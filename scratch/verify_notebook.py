"""Verify the modified 06_emotion_live_pipeline.ipynb: valid JSON + print the
modified cells so the changes can be eyeballed."""
import json
from pathlib import Path

NB = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\06_emotion_live_pipeline.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
print(f"JSON_OK: {len(nb['cells'])} cells")

markers = [
    "# 9. Install PersonaPlex moshi package -- ONLY the copy bundled",
    "# --- 4b. Bundled Moshi package from the bnb4 snapshot",
    "# Hard guard: the moshi package the server will ACTUALLY resolve",
    '"PERSONAPLEX_DIR":     PERSONAPLEX_DIR',
]
for m in markers:
    hits = [i for i, c in enumerate(nb["cells"]) if m in "".join(c.get("source", []))]
    print(f"marker {'FOUND' if hits else 'MISSING!!'} in cell(s) {hits}: {m[:60]}")
    for i in hits[:1]:
        print("=" * 78)
        print(f"--- cell[{i}] source ---")
        print("".join(nb["cells"][i]["source"]))
