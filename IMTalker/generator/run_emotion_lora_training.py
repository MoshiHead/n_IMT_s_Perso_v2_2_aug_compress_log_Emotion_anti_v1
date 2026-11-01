"""
IMTalker Emotion LoRA Training — launcher script.

Trains the emotion conditioning adapters on top of a pretrained IMTalker
generator checkpoint. The base generator is frozen; only the emotion
conditioner module and targeted LoRA adapters are trainable.

Usage (from the training notebook):
    python generator/run_emotion_lora_training.py \
        --repo_root /workspace/speech2avatar/IMTalker \
        --dataset_path /workspace/emotion_dataset \
        --pretrained_ckpt /path/to/generator.ckpt \
        --exp_root /workspace/emotion_exps \
        --exp_name emotion_lora_v1 \
        --wav2vec_model_path /path/to/wav2vec2-base-960h \
        --renderer_ckpt /path/to/renderer.ckpt \
        --lora_rank 32 \
        --lora_alpha 64.0 \
        --batch_size 8 \
        --lr 1e-4 \
        --max_steps 100000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

# torch>=2.6 weights_only=True breaks PL 2.2.1 checkpoint loading
_TORCH_LOAD = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_compat


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="IMTalker emotion LoRA training")
    p.add_argument("--repo_root",           required=True)
    p.add_argument("--dataset_path",        required=True)
    p.add_argument("--base_generator_ckpt", default=None,
                   help="Path to full pretrained base generator checkpoint (generator.ckpt)")
    p.add_argument("--style_lora_ckpt",     default=None,
                   help="Optional pretrained style LoRA checkpoint (e.g. ditto_blink_lora) to merge into base")
    p.add_argument("--style_lora_rank",     type=int, default=64)
    p.add_argument("--style_lora_alpha",    type=float, default=128.0)
    p.add_argument("--pretrained_ckpt",     default=None,
                   help="Legacy alias for base_generator_ckpt or style_lora_ckpt")
    p.add_argument("--exp_root",            required=True)
    p.add_argument("--exp_name",            required=True)
    p.add_argument("--wav2vec_model_path",  required=True)
    p.add_argument("--renderer_ckpt",       required=True)
    # LoRA hyper-parameters
    p.add_argument("--lora_rank",     type=int,   default=32)
    p.add_argument("--lora_alpha",    type=float, default=64.0)
    p.add_argument("--lora_dropout",  type=float, default=0.05)
    # Training hyper-parameters
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--max_steps",     type=int,   default=100_000)
    p.add_argument("--max_epochs",    type=int,   default=-1)
    p.add_argument("--save_freq",     type=int,   default=5_000)
    p.add_argument("--display_freq",  type=int,   default=2_000)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--val_split",     type=int,   default=100,
                   help="Number of samples held out for validation")
    p.add_argument("--resume_ckpt",   default=None,
                   help="Explicit emotion LoRA checkpoint to resume from")
    p.add_argument("--no_auto_resume", action="store_true")
    # Per-epoch test video
    p.add_argument("--test_ref",      default=None)
    p.add_argument("--test_aud",      default=None)
    p.add_argument("--test_out_dir",  default=None)
    p.add_argument("--test_every_n_epochs", type=int, default=1)
    p.add_argument("--test_crop",     type=int, default=0,
                   help="0=do not crop (safe for robot faces); 1=auto-crop")
    p.add_argument("--test_cfg",      type=float, default=3.0)
    p.add_argument("--test_nfe",      type=int, default=10)
    p.add_argument("--test_seed",     type=int, default=42)
    p.add_argument("--test_emotion_id", type=int, default=1,
                   help="Emotion ID to demonstrate in the test video (default: 1 = happy)")
    # Emotion-specific
    p.add_argument("--emotion_filter", nargs="*", default=None,
                   help="Only train on specific emotions e.g. --emotion_filter happy sad")
    p.add_argument("--list_path",     default=None,
                   help="Explicit train list file (one stem per line)")
    p.add_argument("--val_list_path", default=None,
                   help="Explicit val list file")
    return p.parse_args(argv)


# ── Key-rename adapter (same as run_generator_training.py) ──────────────────

class _KeyRenameDataset(torch.utils.data.Dataset):
    """dataset.py returns 'gaze_now' but FM.py reads 'gaze'; fix that."""
    RENAME = (("gaze_now", "gaze"), ("pose_now", "pose"), ("cam_now", "cam"))

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        b = self.ds[i]
        for k_from, k_to in self.RENAME:
            if k_from in b:
                b[k_to] = b[k_from]
        return b


# ── Lightning system ─────────────────────────────────────────────────────────

def _build_system(opt, base_generator_ckpt: Path, style_lora_ckpt: Path,
                  lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Build the LightningModule with frozen base + emotion LoRA."""
    import pytorch_lightning as pl
    from torch import nn, optim
    from generator.FM import FMGenerator
    from generator.train import EMA, append_dims, L1loss
    from generator.train_lora import apply_emotion_lora_to_model, _get_child

    class EmotionLoRASystem(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = FMGenerator(opt)
            self.opt   = opt
            self.loss_fn = L1loss()

            # 1. Load pretrained base generator weights (generator.ckpt)
            base_path = Path(base_generator_ckpt)
            assert base_path.exists(), f"Base generator checkpoint not found: {base_path}"
            b_ckpt = torch.load(str(base_path), map_location="cpu", weights_only=False)
            b_raw  = (b_ckpt.get("ema_state_dict")
                     or b_ckpt.get("state_dict", b_ckpt.get("model", b_ckpt)))
            if isinstance(b_raw, dict) and "model" in b_raw and isinstance(b_raw["model"], dict):
                b_raw = b_raw["model"]
            b_state = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in b_raw.items()
            }
            missing, unexpected = self.model.load_state_dict(b_state, strict=False)
            matched = len(set(self.model.state_dict().keys()) - set(missing))
            print(f"[EmotionLoRA] Base generator loaded from {base_path.name} | "
                  f"{matched} keys matched (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
            if matched < 100:
                raise RuntimeError(
                    f"CRITICAL ERROR: Only {matched} keys matched from base checkpoint {base_path}!\n"
                    f"Ensure base_generator_ckpt points to the full generator.ckpt, NOT a LoRA checkpoint."
                )

            # 2. Optionally merge style LoRA (e.g. ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt)
            if style_lora_ckpt and Path(style_lora_ckpt).exists():
                s_path = Path(style_lora_ckpt)
                s_ckpt = torch.load(str(s_path), map_location="cpu", weights_only=False)
                s_raw  = (s_ckpt.get("ema_state_dict")
                         or s_ckpt.get("state_dict", s_ckpt.get("model", s_ckpt)))
                if isinstance(s_raw, dict) and "model" in s_raw and isinstance(s_raw["model"], dict):
                    s_raw = s_raw["model"]
                s_state = {
                    k.replace("model.", "", 1) if k.startswith("model.") else k: v
                    for k, v in s_raw.items()
                }
                lora_targets = set()
                for k in s_state.keys():
                    if ".lora_A.weight" in k:
                        lora_targets.add(k.replace(".lora_A.weight", ""))
                
                if lora_targets:
                    style_scale = float(getattr(opt, "style_lora_alpha", 128.0)) / max(1, int(getattr(opt, "style_lora_rank", 64)))
                    merged = 0
                    for target in lora_targets:
                        A = s_state.get(f"{target}.lora_A.weight")
                        B = s_state.get(f"{target}.lora_B.weight")
                        if A is not None and B is not None:
                            delta_W = style_scale * (B @ A)
                            mod = _get_child(self.model, target)
                            mod.weight.data += delta_W.to(mod.weight.device)
                            merged += 1
                    print(f"[EmotionLoRA] Merged {merged} style LoRA layers from {s_path.name} (scale={style_scale:.2f}) into base generator weights", flush=True)

            # 3. Apply emotion LoRA (freezes base, adds adapters, unfreezes emotion_conditioner)
            apply_emotion_lora_to_model(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

            # 4. EMA on trainable params only
            self.ema = EMA(self.model, decay=0.9999)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.ema.update()

        def on_validation_epoch_start(self):
            self.ema.apply_shadow()

        def on_validation_epoch_end(self):
            self.ema.restore()

        def on_save_checkpoint(self, checkpoint):
            checkpoint["ema_state_dict"] = self.ema.shadow

        def on_load_checkpoint(self, checkpoint):
            if "ema_state_dict" in checkpoint:
                self.ema.shadow = checkpoint["ema_state_dict"]
                print("[EmotionLoRA] EMA state restored from checkpoint", flush=True)

        def _shared_step(self, batch):
            m_now = batch["m_now"]
            noise  = torch.randn_like(m_now)
            times  = torch.rand(m_now.size(0), device=self.device)
            t      = append_dims(times, m_now.ndim - 1)
            noised = t * m_now + (1 - t) * noise
            gt_flow = m_now - noise
            batch["m_now"] = noised
            pred = self.model(batch, t=times)
            fm_loss  = self.loss_fn(pred, gt_flow)
            vel_loss = self.loss_fn(
                pred[:, 1:] - pred[:, :-1],
                gt_flow[:, 1:] - gt_flow[:, :-1],
            )
            return fm_loss + vel_loss, fm_loss

        def training_step(self, batch, batch_idx):
            loss, fm = self._shared_step(batch)
            self.log("train_loss", loss, prog_bar=True)
            self.log("fm_loss",    fm,   prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            loss, fm = self._shared_step(batch)
            self.log("val_loss",    loss, prog_bar=True)
            self.log("val_fm_loss", fm,   prog_bar=True)

        def configure_optimizers(self):
            emo_params  = list(self.model.emotion_conditioner.parameters())
            lora_params = [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "emotion_conditioner" not in n]
            optimizer = optim.AdamW([
                {"params": emo_params,  "lr": self.opt.lr * 2.0},
                {"params": lora_params, "lr": self.opt.lr},
            ], betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.opt.iter, eta_min=1e-6
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    return EmotionLoRASystem()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    repo_root = Path(args.repo_root).resolve()
    gen_dir   = repo_root / "generator"
    for p in (str(repo_root), str(gen_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from generator.emotion_dataset import EmotionAudioMotionDataset, NUM_EMOTIONS

    # Build opt (reuse TrainOptions structure for all model hyper-params)
    import argparse as _ap
    from generator.train import TrainOptions
    parser = _ap.ArgumentParser()
    parser = TrainOptions().initialize(parser)
    opt = parser.parse_args([])
    opt.dataset_path      = args.dataset_path
    opt.batch_size        = args.batch_size
    opt.lr                = args.lr
    opt.iter              = args.max_steps
    opt.save_freq         = args.save_freq
    opt.display_freq      = args.display_freq
    opt.rank              = "cuda"
    opt.wav2vec_model_path = args.wav2vec_model_path
    opt.fix_noise_seed    = True
    opt.seed              = args.test_seed
    opt.skip_fm_audio_encoder = True   # dataset has precomputed .npy features

    exp_root = Path(args.exp_root).resolve()
    exp_dir  = exp_root / args.exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    opt.exp_path = str(exp_root)
    opt.exp_name = args.exp_name

    # ── Checkpoint Resolution ────────────────────────────────────────────────
    base_generator_ckpt = getattr(args, "base_generator_ckpt", None)
    style_lora_ckpt     = getattr(args, "style_lora_ckpt", None)
    pretrained_ckpt     = getattr(args, "pretrained_ckpt", None)

    if not base_generator_ckpt:
        if pretrained_ckpt:
            p_path = Path(pretrained_ckpt)
            if p_path.exists():
                p_raw = torch.load(str(p_path), map_location="cpu", weights_only=False)
                p_state = p_raw.get("ema_state_dict") or p_raw.get("state_dict", p_raw)
                if isinstance(p_state, dict) and "model" in p_state and isinstance(p_state["model"], dict):
                    p_state = p_state["model"]
                has_lora_keys = any("lora_" in k for k in p_state.keys())
                has_base_keys = any("fmt.blocks.0.attn.qkv.weight" in k for k in p_state.keys())
                
                if has_lora_keys and not has_base_keys:
                    # It's a LoRA checkpoint!
                    style_lora_ckpt = p_path
                    for cand in [
                        repo_root / "checkpoints" / "generator.ckpt",
                        repo_root.parent / "checkpoints" / "generator.ckpt",
                        Path("/workspace/speech2avatar/IMTalker/checkpoints/generator.ckpt"),
                    ]:
                        if cand.exists():
                            base_generator_ckpt = cand
                            break
                    print(f"[Pretrained] Note: {p_path.name} is a LoRA checkpoint. Auto-paired with base generator: {base_generator_ckpt}", flush=True)
                else:
                    base_generator_ckpt = p_path

        if not base_generator_ckpt:
            for cand in [
                repo_root / "checkpoints" / "generator.ckpt",
                repo_root.parent / "checkpoints" / "generator.ckpt",
            ]:
                if cand.exists():
                    base_generator_ckpt = cand
                    break

    assert base_generator_ckpt and Path(base_generator_ckpt).exists(), \
        f"Base generator checkpoint (generator.ckpt) not found: {base_generator_ckpt}\n" \
        "Please provide --base_generator_ckpt pointing to generator.ckpt"

    print(f"[Pretrained] Base generator: {base_generator_ckpt}", flush=True)
    if style_lora_ckpt:
        print(f"[Pretrained] Style LoRA to merge: {style_lora_ckpt}", flush=True)

    opt.style_lora_rank  = getattr(args, "style_lora_rank", 64)
    opt.style_lora_alpha = getattr(args, "style_lora_alpha", 128.0)

    # ── Datasets ─────────────────────────────────────────────────────────────
    all_motion = sorted(Path(args.dataset_path).glob("motion/*.pt"))
    n_total    = len(all_motion)
    val_count  = args.val_split if n_total > args.val_split + 10 else max(1, n_total // 10)
    print(f"[Data] {n_total} clips -> train={n_total - val_count} val={val_count}", flush=True)

    ds_train = _KeyRenameDataset(EmotionAudioMotionDataset(
        opt=opt, start=0, end=-val_count, split="train",
        list_path=args.list_path,
        emotion_filter=args.emotion_filter,
    ))
    ds_val = _KeyRenameDataset(EmotionAudioMotionDataset(
        opt=opt, start=-val_count, end=None, split="val",
        list_path=args.val_list_path,
        emotion_filter=args.emotion_filter,
    ))

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=opt.batch_size, shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=min(8, opt.batch_size), shuffle=False,
        num_workers=0,
    )

    # ── System ───────────────────────────────────────────────────────────────
    system = _build_system(
        opt=opt,
        base_generator_ckpt=base_generator_ckpt,
        style_lora_ckpt=style_lora_ckpt,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # ── Resume logic ─────────────────────────────────────────────────────────
    ckpt_path = None
    if args.resume_ckpt and Path(args.resume_ckpt).exists():
        ckpt_path = args.resume_ckpt
    elif not args.no_auto_resume:
        last = exp_dir / "checkpoints" / "last.ckpt"
        if last.exists():
            ckpt_path = str(last)
    if ckpt_path:
        print(f"[Resume] {ckpt_path}", flush=True)

    # ── Callbacks / loggers ──────────────────────────────────────────────────
    tb_logger  = TensorBoardLogger(save_dir=str(exp_root), name=args.exp_name)
    csv_logger = CSVLogger(save_dir=str(exp_root), name=args.exp_name)

    # Text log callback (reuse from imtalker_train_utils)
    text_log = None
    try:
        try:
            from generator.imtalker_train_utils import TextLogCallback
        except ImportError:
            for sp in (str(repo_root / "preprocessing"), str(repo_root / "generator")):
                if sp not in sys.path:
                    sys.path.insert(0, sp)
            from imtalker_train_utils import TextLogCallback
        text_log = TextLogCallback(exp_dir / "train_log.txt", every_n_steps=args.log_every_n_steps)
    except Exception as e:
        print(f"[warn] TextLogCallback not initialized: {e}", flush=True)
        text_log = None

    test_video_cb = None
    if args.test_ref and args.test_aud and Path(args.test_ref).exists() and Path(args.test_aud).exists():
        try:
            try:
                from generator.imtalker_train_utils import EpochTestVideoCallback
            except ImportError:
                from imtalker_train_utils import EpochTestVideoCallback
            test_video_cb = EpochTestVideoCallback(
                ref_path=args.test_ref,
                aud_path=args.test_aud,
                out_dir=args.test_out_dir or str(exp_dir / "test_videos"),
                renderer_ckpt=args.renderer_ckpt,
                wav2vec_model_path=args.wav2vec_model_path,
                every_n_epochs=args.test_every_n_epochs,
                crop=bool(args.test_crop),
                a_cfg_scale=args.test_cfg,
                nfe=args.test_nfe,
                seed=args.test_seed,
            )
            print(f"[Callback] EpochTestVideoCallback enabled | every {args.test_every_n_epochs} epochs", flush=True)
        except Exception as e:
            print(f"[warn] EpochTestVideoCallback not initialized: {e}", flush=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="{step:06d}",
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks  = [checkpoint_callback, lr_monitor]
    if text_log:
        callbacks.append(text_log)
    if test_video_cb:
        callbacks.append(test_video_cb)

    # ── val_check_interval safety ─────────────────────────────────────────────
    n_gpu = max(1, torch.cuda.device_count())
    steps_per_epoch = max(1, math.ceil(len(ds_train) / (opt.batch_size * n_gpu)))
    display_freq    = min(opt.display_freq, steps_per_epoch)

    # ── Trainer ──────────────────────────────────────────────────────────────
    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"
    print(f"[Trainer] gpus={n_gpu} strategy={strategy} max_steps={opt.iter} "
          f"val_check_interval={display_freq}", flush=True)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=strategy,
        max_steps=opt.iter,
        max_epochs=args.max_epochs,
        val_check_interval=display_freq,
        check_val_every_n_epoch=None,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    trainer.fit(system, dl_train, dl_val, ckpt_path=ckpt_path)
    print("[Done] emotion LoRA training finished", flush=True)


if __name__ == "__main__":
    main(parse_args())
