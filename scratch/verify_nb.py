import ast, json
from pathlib import Path

P = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\06_emotion_live_pipeline.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
print("[ok] notebook JSON valid,", len(nb["cells"]), "cells")
bad = 0
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        bad += 1
        print(f"[SYNTAX ERROR] cell {i}: {e}")
if bad == 0:
    print("[ok] all code cells parse")
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "EMOTION_A_CFG_SCALE" in src:
            print(f"[cell {i}] contains EMOTION_A_CFG_SCALE")
        if "launch_env[\"A_CFG_SCALE\"]" in src:
            print(f"[cell {i}] launch env sets A_CFG_SCALE/NFE")
