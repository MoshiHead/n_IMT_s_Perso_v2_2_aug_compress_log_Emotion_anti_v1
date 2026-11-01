"""
Render talking videos across all 6 emotions for a given emotion LoRA checkpoint.

Usage:
    python generator/test_emotions.py \
        --checkpoint /path/to/emotion_exps/emotion_lora_v1/checkpoints/last.ckpt \
        --ref_img assets/3robert.jpeg \
        --audio assets/audio_1.wav \
        --out_dir test_renders \
        --cfg 3.0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

# Fix weights_only in torch >= 2.6
_TORCH_LOAD = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_compat


def parse_args():
    p = argparse.ArgumentParser(description="Render test videos for all 6 emotions")
    p.add_argument("--checkpoint", required=True, help="Trained emotion LoRA checkpoint (.ckpt)")
    p.add_argument("--base_generator_ckpt", default=None, help="Base generator.ckpt (auto-detected if None)")
    p.add_argument("--style_lora_ckpt", default=None, help="Optional live winner style LoRA")
    p.add_argument("--ref_img", default="assets/3robert.jpeg")
    p.add_argument("--audio", default="assets/audio_1.wav")
    p.add_argument("--out_dir", default="test_emotion_renders")
    p.add_argument("--renderer_ckpt", default=None)
    p.add_argument("--wav2vec_model_path", default=None)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=64.0)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--nfe", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--emotions", nargs="*", default=None,
                   help="Specific emotions to test, e.g. --emotions happy sad angry")
    return p.parse_args()


def save_video(vid_tensor: torch.Tensor, video_path: Path, audio_path: str):
    import torchvision
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_filename = tmp.name

    vid = vid_tensor.permute(0, 2, 3, 1).detach().clamp(0, 1).cpu()
    torchvision.io.write_video(temp_filename, (vid * 255).to(torch.uint8), fps=25)

    cmd = (
        f'ffmpeg -y -v warning -i "{temp_filename}" -i "{audio_path}" '
        f'-c:v copy -c:a aac -shortest "{video_path}"'
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    if res.returncode != 0:
        import shutil
        shutil.copy2(temp_filename, video_path)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(repo_root / "generator") not in sys.path:
        sys.path.insert(0, str(repo_root / "generator"))

    from generator.options.base_options import BaseOptions
    from generator.FM import FMGenerator, EMOTION_MAP, EMOTION_NAMES
    from generator.train_lora import apply_emotion_lora_to_model, _get_child
    from generator.generate import DataProcessor, InferenceOptions
    from renderer.models import IMTRenderer
    from generator.wav2vec2 import Wav2VecModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Checkpoint paths
    ckpt_dir = repo_root / "checkpoints"
    renderer_ckpt = Path(args.renderer_ckpt or (ckpt_dir / "renderer.ckpt"))
    w2v_path = Path(args.wav2vec_model_path or (ckpt_dir / "wav2vec2-base-960h"))
    base_gen = Path(args.base_generator_ckpt or (ckpt_dir / "generator.ckpt"))

    assert renderer_ckpt.exists(), f"renderer missing: {renderer_ckpt}"
    assert w2v_path.exists(), f"wav2vec missing: {w2v_path}"
    assert base_gen.exists(), f"base generator missing: {base_gen}"

    # Build model
    parser = argparse.ArgumentParser()
    parser = BaseOptions().initialize(parser)
    parser.add_argument("--skip_fm_audio_encoder", action="store_true")
    opt = parser.parse_args([])
    opt.rank = device
    opt.wav2vec_model_path = str(w2v_path)
    opt.skip_fm_audio_encoder = True
    opt.fix_noise_seed = True
    opt.seed = args.seed

    model = FMGenerator(opt).to(device)

    # 1. Load base weights
    b_ckpt = torch.load(str(base_gen), map_location="cpu")
    b_raw  = b_ckpt.get("ema_state_dict") or b_ckpt.get("state_dict", b_ckpt)
    if "model" in b_raw and isinstance(b_raw["model"], dict): b_raw = b_raw["model"]
    b_state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in b_raw.items()}
    model.load_state_dict(b_state, strict=False)

    # 2. Merge style LoRA if present
    if args.style_lora_ckpt and Path(args.style_lora_ckpt).exists():
        s_path = Path(args.style_lora_ckpt)
        s_ckpt = torch.load(str(s_path), map_location="cpu")
        s_raw  = s_ckpt.get("ema_state_dict") or s_ckpt.get("state_dict", s_ckpt)
        if "model" in s_raw and isinstance(s_raw["model"], dict): s_raw = s_raw["model"]
        s_state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in s_raw.items()}
        lora_targets = {k.replace(".lora_A.weight", "") for k in s_state if ".lora_A.weight" in k}
        style_scale = 128.0 / 64.0
        for target in lora_targets:
            A = s_state.get(f"{target}.lora_A.weight")
            B = s_state.get(f"{target}.lora_B.weight")
            if A is not None and B is not None:
                mod = _get_child(model, target)
                mod.weight.data += (style_scale * (B @ A)).to(mod.weight.device)
        print(f"Merged {len(lora_targets)} style LoRA layers")

    # 3. Apply emotion LoRA
    apply_emotion_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=0.0)

    # 4. Load trained emotion checkpoint
    e_ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    e_raw  = e_ckpt.get("ema_state_dict") or e_ckpt.get("state_dict", e_ckpt)
    if "model" in e_raw and isinstance(e_raw["model"], dict): e_raw = e_raw["model"]
    e_state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in e_raw.items()}
    missing, unexpected = model.load_state_dict(e_state, strict=False)
    print(f"Loaded emotion checkpoint: {Path(args.checkpoint).name} (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # Load Renderer & Wav2Vec
    p_inf = argparse.ArgumentParser()
    p_inf = InferenceOptions().initialize(p_inf)
    opt_inf = p_inf.parse_args([])
    opt_inf.rank = device
    opt_inf.renderer_path = str(renderer_ckpt)
    opt_inf.wav2vec_model_path = str(w2v_path)

    renderer = IMTRenderer(opt_inf).to(device).eval()
    r_ckpt = torch.load(str(renderer_ckpt), map_location="cpu")
    r_state = {k.replace("gen.", ""): v for k, v in r_ckpt["state_dict"].items() if k.startswith("gen.")}
    renderer.load_state_dict(r_state, strict=False)

    wav2vec = Wav2VecModel.from_pretrained(str(w2v_path), local_files_only=True).to(device).eval()
    data_processor = DataProcessor(opt_inf)

    # Preprocess reference image and audio
    ref_path = Path(args.ref_img)
    if not ref_path.is_absolute():
        ref_path = repo_root / ref_path
    aud_path = Path(args.audio)
    if not aud_path.is_absolute():
        aud_path = repo_root / aud_path

    print(f"Reference image: {ref_path}")
    print(f"Audio file:      {aud_path}")

    data = data_processor.preprocess(str(ref_path), str(aud_path), crop=False)
    s = data["s"].to(device)
    f_r, g_r = renderer.dense_feature_encoder(s)
    t_r = renderer.latent_token_encoder(s)
    data["ref_x"] = t_r
    data["pose"] = None
    data["cam"] = None
    data["gaze"] = None

    # Wav2Vec features
    raw_audio = data["a"].squeeze(0).to(device)
    T_target = math.ceil(raw_audio.shape[0] * 25 / 16000)
    target_samples = int(T_target * 16000 / 25)
    if raw_audio.shape[0] < target_samples:
        diff = target_samples - raw_audio.shape[0]
        raw_audio = F.pad(raw_audio.unsqueeze(0), (0, diff), mode="replicate").squeeze(0)
    else:
        raw_audio = raw_audio[:target_samples]
    with torch.no_grad():
        out = wav2vec(raw_audio.unsqueeze(0), seq_len=T_target, output_hidden_states=False)
        feat_25 = out.last_hidden_state
    data["a_feat"] = feat_25

    # Determine which emotions to render
    if args.emotions:
        test_eids = []
        for e in args.emotions:
            e_clean = e.strip().lower()
            if e_clean.isdigit():
                test_eids.append(int(e_clean))
            elif e_clean in EMOTION_MAP:
                test_eids.append(EMOTION_MAP[e_clean])
    else:
        test_eids = [0, 1, 2, 3, 4, 5]  # all 6 emotions

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating videos for emotions: {[EMOTION_NAMES.get(e, e) for e in test_eids]} (cfg={args.cfg}) ...")

    for eid in test_eids:
        emo_name = EMOTION_NAMES.get(eid, f"emotion_{eid}")
        print(f"  Rendering [{eid}] {emo_name} ...", flush=True)
        data["emotion_id"] = eid

        with torch.no_grad():
            sample = model.sample(data, a_cfg_scale=args.cfg, nfe=args.nfe, seed=args.seed)

            # Decode
            T = sample.shape[1]
            ta_r = renderer.adapt(t_r, g_r)
            m_r = renderer.latent_token_decoder(ta_r)
            d_hat = []
            for t in range(T):
                ta_c = renderer.adapt(sample[:, t, ...], g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                d_i = renderer.decode(m_c, m_r, f_r)
                d_hat.append(d_i)

            vid_tensor = torch.cat(d_hat, dim=0)
            out_file = out_dir / f"{eid}_{emo_name}.mp4"
            save_video(vid_tensor, out_file, str(aud_path))
            print(f"    ✓ Saved -> {out_file}")

    print(f"\nAll renders saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
