import json
from pathlib import Path

root = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1")
for tag, p in (("NEW", root / "06_emotion_live_pipeline.ipynb"), ("BAK", root / "scratch" / "backup_pre_lagfix" / "06_emotion_live_pipeline.ipynb")):
    nb = json.loads(p.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][1].get("source", []))
    print(f"--- [{tag}] cell 1 (first 6 lines) ---")
    for ln in src.splitlines()[:6]:
        print("  " + ln)
