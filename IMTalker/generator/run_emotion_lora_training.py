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
    p.add_argument("--repo_root",         required=True)
    p.add_argument("--dataset_path",      required=True)
    p.add_argument("--pretrained_ckpt",   required=True,
                   help="Path to pretrained base generator checkpoint (.ckpt)")
    p.add_argument("--exp_root",          required=True)
    p.add_argument("--exp_name",          required=True)
    p.add_argument("--wav2vec_model_path", required=True)
    p.add_argument("--renderer_ckpt",     required=True)
    # LoRA hyper-parameters
    p.add_argument("--lora_rank",   type=int,   default=32)
    p.add_argument("--lora_alpha",  type=float, default=64.0)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    # Training hyper-parameters
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--max_steps",   type=int,   default=100_000)
    p.add_argument("--max_epochs",  type=int,   default=-1)
    p.add_argument("--save_freq",   type=int,   default=5_000)
    p.add_argument("--display_freq",type=int,   default=2_000)
    p.add_argument("--log_every_n_steps", type=int, default=50)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_split",   type=int,   default=100,
                   help="Number of samples held out for validation")
    p.add_argument("--resume_ckpt", default=None,
                   help="Explicit emotion LoRA checkpoint to resume from")
    p.add_argument("--no_auto_resume", action="store_true")
    # Per-epoch test video
    p.add_argument("--test_ref",    default=None)
    p.add_argument("--test_aud",    default=None)
    p.add_argument("--test_out_dir",default=None)
    p.add_argument("--test_every_n_epochs", type=int, default=1)
    p.add_argument("--test_crop",   type=int, default=1)
    p.add_argument("--test_cfg",    type=float, default=3.0)
    p.add_argument("--test_nfe",    type=int, default=10)
    p.add_argument("--test_seed",   type=int, default=42)
    # Emotion-specific
    p.add_argument("--emotion_filter", nargs="*", default=None,
                   help="Only train on specific emotions e.g. --emotion_filter happy sad")
    p.add_argument("--list_path",   default=None,
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

def _build_system(opt, pretrained_ckpt: Path, lora_rank: int,
                  lora_alpha: float, lora_dropout: float):
    """Build the LightningModule with frozen base + emotion LoRA."""
    import pytorch_lightning as pl
    from torch import nn, optim
    from generator.FM import FMGenerator
    from generator.train import EMA, append_dims, L1loss
    from generator.train_lora import apply_emotion_lora_to_model

    class EmotionLoRASystem(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = FMGenerator(opt)
            self.opt   = opt
            self.loss_fn = L1loss()

            # 1. Load pretrained base weights (strict=False: emotion_conditioner is new)
            ckpt = torch.load(str(pretrained_ckpt), map_location="cpu")
            raw  = (ckpt.get("ema_state_dict")
                    or ckpt.get("state_dict", ckpt.get("model", ckpt)))
            if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
                raw = raw["model"]
            state = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in raw.items()
            }
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            new_keys = [k for k in missing if "emotion_conditioner" in k]
            other_missing = [k for k in missing if "emotion_conditioner" not in k]
            print(f"[EmotionLoRA] pretrained loaded | "
                  f"new_keys={len(new_keys)} other_missing={len(other_missing)} "
                  f"unexpected={len(unexpected)}", flush=True)

            # 2. Apply emotion LoRA (freezes base, adds adapters, unfreezes emotion_conditioner)
            apply_emotion_lora_to_model(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

            # 3. EMA on trainable params only
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

    pretrained_ckpt = Path(args.pretrained_ckpt)
    assert pretrained_ckpt.exists(), f"pretrained_ckpt not found: {pretrained_ckpt}"

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
        pretrained_ckpt=pretrained_ckpt,
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
    try:
        scripts_dir = repo_root / "preprocessing"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from imtalker_train_utils import TextLogCallback
        text_log = TextLogCallback(exp_dir / "train_log.txt", every_n_steps=args.log_every_n_steps)
    except ImportError:
        text_log = None

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
