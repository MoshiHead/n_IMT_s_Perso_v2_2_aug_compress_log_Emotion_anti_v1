"""fix6: notebook -- add EMOTION_A_CFG_SCALE/EMOTION_NFE parameters and pass
them to the live server launch env when the emotion LoRA is active."""
import json
from pathlib import Path

P = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\06_emotion_live_pipeline.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

param_anchor = "EMOTION_LORA_ALPHA= 64.0"
launch_anchor = "print(f\"Launching emotion-aware live server"
update_anchor = "launch_env.update({"

param_done = launch_done = False

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", [])
    joined = "".join(src)

    if not param_done and param_anchor in joined and "EMOTION_A_CFG_SCALE" not in joined:
        for idx, line in enumerate(src):
            if line.startswith(param_anchor):
                new_lines = [
                    "\n",
                    "# Inference settings for the emotion LoRA. These MUST match the offline\n",
                    "# validation in IMTalker/generator/test_emotions.py, which the emotion LoRA\n",
                    "# was tuned and verified with (a_cfg_scale=3.0, nfe=10): a_cfg_scale is the\n",
                    "# audio classifier-free-guidance scale -- it controls how tightly the\n",
                    "# generated motion follows the audio, so a low value weakens lip-sync. The\n",
                    "# live-winner defaults (1.13 / 5) were tuned for the pre-emotion blink LoRA\n",
                    "# and are only used when the emotion LoRA is NOT active.\n",
                    "EMOTION_A_CFG_SCALE = 3.0   # audio CFG: how strongly motion follows audio\n",
                    "EMOTION_NFE = 10            # FM denoising steps (offline test default)\n",
                ]
                src[idx + 1:idx + 1] = new_lines
                cell["source"] = src
                param_done = True
                print("[nb] PARAMETERS cell: added EMOTION_A_CFG_SCALE / EMOTION_NFE")
                break

    if not launch_done and update_anchor in joined and launch_anchor in joined and "A_CFG_SCALE" not in joined:
        for idx, line in enumerate(src):
            if line.startswith(launch_anchor):
                new_lines = [
                    "# Emotion-LoRA inference settings: match the offline test_emotions.py\n",
                    "# validation (a_cfg_scale=3.0, nfe=10) so streaming lip-sync matches the\n",
                    "# offline model. Without the emotion LoRA, start_winner_live.sh falls back\n",
                    "# to its own defaults (1.13 / 5), tuned for the blink-LoRA winner.\n",
                    "if _HAS_EMOTION_LORA:\n",
                    "    launch_env[\"A_CFG_SCALE\"] = str(EMOTION_A_CFG_SCALE)\n",
                    "    launch_env[\"NFE\"] = str(EMOTION_NFE)\n",
                    "\n",
                ]
                src[idx:idx] = new_lines
                cell["source"] = src
                launch_done = True
                print("[nb] LAUNCH cell: pass A_CFG_SCALE / NFE when emotion LoRA active")
                break

assert param_done, "PARAMETERS cell anchor not found"
assert launch_done, "LAUNCH cell anchor not found"

with open(P, "w", encoding="utf-8", newline="") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("[ok] notebook updated")
