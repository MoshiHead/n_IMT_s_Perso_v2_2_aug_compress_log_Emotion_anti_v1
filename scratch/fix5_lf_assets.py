"""fix5: restore LF line endings on edited files (bash breaks on CRLF) and
finish the download_live_assets.sh search removal."""
from pathlib import Path

ROOT = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1")


def write_lf(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def normalize_lf(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        src = fh.read()
    fixed = src.replace("\r\n", "\n")
    if fixed != src:
        write_lf(path, fixed)
        print(f"[lf] normalized {path.name}")
    else:
        print(f"[lf] already LF: {path.name}")


for rel in (
    "IMTalker/start_winner_live.sh",
    "run_live.sh",
    "IMTalker/liveTry.py",
    "IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py",
):
    normalize_lf(ROOT / rel)

# ---- finish scripts/download_live_assets.sh ----
P = ROOT / "scripts" / "download_live_assets.sh"
with open(P, "r", encoding="utf-8", newline="") as fh:
    src = fh.read()
src = src.replace("\r\n", "\n")
orig = src

i = src.find("# --- Reference-LoRA addition")
assert i >= 0, "ref-lora block start not found"
j = src.find("# --- Emotion LoRA adapter", i)
assert j >= 0, "emotion block start not found"
src = src[:i] + src[j:]
print(f"[assets] removed ref-lora block ({j - i} chars)")

old = 'echo "[12/12] Downloading emotion LoRA adapter (bitwisemind/emotion_lora_adapter_v1)..."'
assert src.count(old) == 1
src = src.replace(old, old.replace("[12/12]", "[11/11]"), 1)
print("[assets] renumbered emotion step to [11/11]")

old_echo = '\necho "  reference LoRA (optional): $REF_LORA_DIR/lora"'
assert src.count(old_echo) == 1, f"echo count={src.count(old_echo)}"
src = src.replace(old_echo, "", 1)
print("[assets] removed ref-lora summary echo")

assert src != orig
write_lf(P, src)
print("[ok] download_live_assets.sh updated (LF)")
