"""fix4: remove search machinery from the shell scripts; render_sub_batch 8 -> 24."""
from pathlib import Path

ROOT = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1")


def region(text, start, end, new, desc):
    i = text.find(start)
    assert i >= 0, f"start not found: {desc}"
    j = text.find(end, i + len(start))
    assert j >= 0, f"end not found: {desc}"
    print(f"[{desc}] removed {j - i} chars")
    return text[:i] + new + text[j:]


def replace_once(text, old, new, desc):
    assert text.count(old) == 1, f"anchor not unique/found ({desc}): count={text.count(old)}"
    print(f"[edit] {desc}")
    return text.replace(old, new, 1)


# ---- start_winner_live.sh ----
P = ROOT / "IMTalker" / "start_winner_live.sh"
src = P.read_text(encoding="utf-8")
orig = src

src = region(
    src,
    "# STT + query routing + web search -- opt-in, additive. ENABLE_SEARCH=0",
    "GENERATOR_PATH=",
    '''# Per-session conversation logs (opt-in): set CONVERSATION_LOG_DIR to get
# conversation_<session>.log/.jsonl + detailed_<session>.log per server run.
# Empty disables file logging (console prints always happen).
LOG_ARGS=()
if [[ -n "${CONVERSATION_LOG_DIR:-}" ]]; then
  LOG_ARGS=(--conversation_log_dir "$CONVERSATION_LOG_DIR")
fi

''',
    "winner: search block -> log args",
)

src = region(
    src,
    '''if [[ "$ENABLE_SEARCH" == "1" ]]; then
  for required in \\''',
    "if [[ -z \"${VOICE_PROMPT_DIR:-}\" ]]; then",
    "",
    "winner: required-path check",
)

src = replace_once(
    src,
    "  --render_sub_batch 8 \\",
    "  --render_sub_batch \"${RENDER_SUB_BATCH:-24}\" \\",
    "winner: render_sub_batch 24",
)

src = replace_once(
    src,
    '''  "${EMOTION_ARGS[@]}" \\
  "${SEARCH_ARGS[@]}"''',
    '''  "${EMOTION_ARGS[@]}" \\
  "${LOG_ARGS[@]}"''',
    "winner: exec args",
)

assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] start_winner_live.sh updated")

# ---- run_live.sh ----
P = ROOT / "run_live.sh"
src = P.read_text(encoding="utf-8")
orig = src
src = region(
    src,
    '''  ENABLE_SEARCH="${ENABLE_SEARCH:-${ENABLE_RAG:-0}}" \\''',
    '''  bash "$IMTALKER_DIR/start_typeah_audio_pace_varm3.sh"''',
    "",
    "run_live: search env lines",
)
assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] run_live.sh updated")

# ---- scripts/download_live_assets.sh ----
P = ROOT / "scripts" / "download_live_assets.sh"
src = P.read_text(encoding="utf-8")
orig = src
src = region(
    src,
    "# --- Reference-LoRA addition",
    "# --- Emotion LoRA adapter",
    "",
    "assets: ref-lora block",
)
src = replace_once(
    src,
    'echo "[12/12] Downloading emotion LoRA adapter (bitwisemind/emotion_lora_adapter_v1)..."',
    'echo "[11/11] Downloading emotion LoRA adapter (bitwisemind/emotion_lora_adapter_v1)..."',
    "assets: renumber emotion step",
)
src = region(
    src,
    '''echo "  reference LoRA (optional): $REF_LORA_DIR/lora"''',
    "echo",
    "",
    "assets: ref-lora echo",
)
assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] download_live_assets.sh updated")
