"""
Training utilities for IMTalker Generator and Emotion LoRA training.

Contains two Lightning callbacks:
  - TextLogCallback:        appends human-readable progress (epoch / step /
                            loss / lr) to a plain-text log file on rank 0.
  - EpochTestVideoCallback: after every N training epochs, renders a test video
                            using the live in-memory generator weights and
                            the IMTRenderer. Rank 0 only.
"""
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class TextLogCallback(pl.Callback):
    """Writes epoch / step / loss / lr lines to a plain-text log file (rank 0)."""

    def __init__(self, log_file, every_n_steps=50):
        super().__init__()
        self.log_file = Path(log_file)
        self.every_n_steps = max(1, int(every_n_steps))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, line):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self._write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] training started | "
            f"gpus={trainer.num_devices} | max_steps={trainer.max_steps} | "
            f"max_epochs={trainer.max_epochs}"
        )

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        loss = float(trainer.callback_metrics.get("train_loss", float("nan")))
        fm = float(trainer.callback_metrics.get("fm_loss", float("nan")))
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch={trainer.current_epoch:04d} "
            f"step={trainer.global_step:07d} lr={lr:.3e} "
            f"train_loss={loss:.6f} fm_loss={fm:.6f}"
        )

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] VALIDATION epoch={trainer.current_epoch:04d} "
            f"step={trainer.global_step:07d} "
            f"val_loss={float(m.get('val_loss', float('nan'))):.6f} "
            f"val_fm_loss={float(m.get('val_fm_loss', float('nan'))):.6f}"
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self._write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch={trainer.current_epoch:04d} "
            f"completed at step={trainer.global_step:07d}"
        )


class EpochTestVideoCallback(pl.Callback):
    """
    Renders a test video after every N training epochs directly with the current
    generator weights (pl_module.model) and IMTRenderer. Rank 0 only.
    """

    def __init__(self, ref_path, aud_path, out_dir, renderer_ckpt, wav2vec_model_path,
                 every_n_epochs=1, crop=False, a_cfg_scale=3.0, nfe=10, seed=42, test_emotion_id=1):
        super().__init__()
        self.ref_path = str(ref_path) if ref_path else None
        self.aud_path = str(aud_path) if aud_path else None
        self.out_dir = Path(out_dir)
        self.renderer_ckpt = str(renderer_ckpt)
        self.wav2vec_model_path = str(wav2vec_model_path)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.crop = bool(crop)
        self.a_cfg_scale = float(a_cfg_scale)
        self.nfe = int(nfe)
        self.seed = int(seed)
        self.test_emotion_id = int(test_emotion_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = None
        self.wav2vec = None
        self.data_processor = None
        self.opt = None

    def _init_components(self, device):
        if self.renderer is not None:
            return

        import argparse
        from renderer.models import IMTRenderer
        from generator.generate import DataProcessor, InferenceOptions
        from generator.wav2vec2 import Wav2VecModel

        parser = argparse.ArgumentParser()
        parser = InferenceOptions().initialize(parser)
        opt = parser.parse_args([])
        opt.rank = device
        opt.renderer_path = self.renderer_ckpt
        opt.wav2vec_model_path = self.wav2vec_model_path
        self.opt = opt

        # 1. Renderer
        self.renderer = IMTRenderer(opt).to(device).eval()
        r_ckpt = torch.load(self.renderer_ckpt, map_location="cpu", weights_only=False)
        r_state = r_ckpt.get("state_dict", r_ckpt)
        ae_state = {k.replace("gen.", ""): v for k, v in r_state.items() if k.startswith("gen.")}
        self.renderer.load_state_dict(ae_state, strict=False)
        print(f"[EpochTest] Renderer loaded from {Path(self.renderer_ckpt).name}", flush=True)

        # 2. Wav2Vec for audio feature extraction
        self.wav2vec = Wav2VecModel.from_pretrained(self.wav2vec_model_path, local_files_only=True).to(device).eval()
        self.wav2vec.feature_extractor._freeze_parameters()
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        print(f"[EpochTest] Wav2Vec2 loaded from {Path(self.wav2vec_model_path).name}", flush=True)

        # 3. DataProcessor
        self.data_processor = DataProcessor(opt)

    def _extract_audio_features(self, speech_tensor, device, fps=25, sr=16000):
        # speech_tensor is [1, samples]
        raw = speech_tensor.squeeze(0).to(device)
        total_samples = raw.shape[0]
        T = int(math.ceil(total_samples * fps / sr))
        target_samples = int(T * sr / fps)
        if raw.shape[0] < target_samples:
            diff = target_samples - raw.shape[0]
            raw = F.pad(raw.unsqueeze(0), (0, diff), mode="replicate").squeeze(0)
        else:
            raw = raw[:target_samples]
        
        with torch.no_grad():
            out = self.wav2vec(raw.unsqueeze(0), seq_len=T, output_hidden_states=False)
            feat = out.last_hidden_state  # [1, T, 768]
        return feat, T

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if not self.ref_path or not Path(self.ref_path).exists():
            return
        if not self.aud_path or not Path(self.aud_path).exists():
            return

        try:
            device = pl_module.device
            self._init_components(device)

            # Apply EMA shadow if present so test video uses clean averaged weights
            if hasattr(pl_module, "ema") and pl_module.ema is not None:
                pl_module.ema.apply_shadow()

            was_training = pl_module.model.training
            pl_module.model.eval()

            with torch.no_grad():
                data = self.data_processor.preprocess(self.ref_path, self.aud_path, crop=self.crop)
                s = data["s"].to(device)
                f_r, g_r = self.renderer.dense_feature_encoder(s)
                t_r = self.renderer.latent_token_encoder(s)
                data["ref_x"] = t_r
                data["emotion_id"] = self.test_emotion_id
                data["pose"] = None
                data["cam"] = None
                data["gaze"] = None

                # Extract audio features directly so skip_fm_audio_encoder works perfectly
                feat_25, T = self._extract_audio_features(data["a"], device)
                data["a_feat"] = feat_25

                from generator.FM import EMOTION_NAMES
                emo_str = EMOTION_NAMES.get(self.test_emotion_id, str(self.test_emotion_id))
                print(f"[EpochTest] epoch {trainer.current_epoch}: generating test video with emotion={emo_str} (id={self.test_emotion_id}) | ref={Path(self.ref_path).name} | a_cfg={self.a_cfg_scale}", flush=True)

                # Sample motion latents with the in-memory trained model
                sample = pl_module.model.sample(
                    data,
                    a_cfg_scale=self.a_cfg_scale,
                    nfe=self.nfe,
                    seed=self.seed,
                )

                # Decode frames with IMTRenderer
                T = sample.shape[1]
                ta_r = self.renderer.adapt(t_r, g_r)
                m_r = self.renderer.latent_token_decoder(ta_r)
                d_hat = []
                for t in range(T):
                    ta_c = self.renderer.adapt(sample[:, t, ...], g_r)
                    m_c = self.renderer.latent_token_decoder(ta_c)
                    d_i = self.renderer.decode(m_c, m_r, f_r)
                    d_hat.append(d_i)

                vid_tensor = torch.cat(d_hat, dim=0)  # [T, 3, 512, 512]
                out_mp4 = self.out_dir / f"epoch_{trainer.current_epoch:04d}.mp4"
                self._save_video(vid_tensor, out_mp4, self.aud_path)
                print(f"[EpochTest] epoch {trainer.current_epoch}: successfully saved test video -> {out_mp4}", flush=True)

            if was_training:
                pl_module.model.train()
            if hasattr(pl_module, "ema") and pl_module.ema is not None:
                pl_module.ema.restore()

        except Exception as e:
            print(f"[EpochTest] epoch {trainer.current_epoch}: FAILED - {e}", flush=True)
            import traceback
            traceback.print_exc()

    def _save_video(self, vid_tensor, video_path, audio_path):
        import tempfile, subprocess, torchvision
        video_path = str(video_path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_filename = tmp.name

        # vid_tensor is [T, 3, 512, 512] with values in [0, 1]
        vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
        vid = (vid * 255).type(torch.ByteTensor)
        torchvision.io.write_video(temp_filename, vid, fps=getattr(self.opt, "fps", 25))

        if audio_path and Path(audio_path).exists():
            cmd = f'ffmpeg -i "{temp_filename}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{video_path}" -y'
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        else:
            if os.path.exists(video_path):
                os.remove(video_path)
            os.rename(temp_filename, video_path)
