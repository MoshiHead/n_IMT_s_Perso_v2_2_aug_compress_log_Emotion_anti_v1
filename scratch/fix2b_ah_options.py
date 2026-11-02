"""AH server search removal, part B: CLI options + engine construction kwargs."""
from pathlib import Path

P = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\IMTalker\liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py")
src = P.read_text(encoding="utf-8")
orig = src


def region(text, start, end, new, desc):
    i = text.find(start)
    assert i >= 0, f"start not found: {desc}"
    j = text.find(end, i + len(start))
    assert j >= 0, f"end not found: {desc}"
    print(f"[{desc}] removed {j - i} chars")
    return text[:i] + new + text[j:]


# L. options: search args block 1 (ref_lora .. stt_pkg_dir), keep --max_input_buffer_sec
src = region(src,
    "        # STT + query routing + web search (all optional; omitting every flag",
    '''        parser.add_argument(
            "--max_input_buffer_sec", type=float, default=2.0,''',
    "",
    "options search args 1")

# M. options: search args block 2 (vad_threshold .. thinking_sound_path), keep --conversation_log_dir
src = region(src,
    '''        parser.add_argument("--vad_threshold", type=float, default=0.5)''',
    '''        parser.add_argument(
            "--conversation_log_dir", default="",''',
    "",
    "options search args 2")

# N. conversation_log_dir help text: drop search mentions
src = region(src,
    '''            help="Directory for structured per-session conversation logs (user transcripts, "''',
    '''                 "timing, errors) as both a .log and a .jsonl file. Empty disables file logging "''',
    '''            help="Directory for structured per-session conversation logs (assistant responses, "
''',
    "log dir help")

# O. get_moshi_engine: drop search kwargs, keep backlog cap + log dir
src = region(src,
    '''                ref_lora_dir=getattr(args, "ref_lora_dir", ""),''',
    '''            )
        return moshi_engine''',
    '''                max_input_buffer_sec=float(getattr(args, "max_input_buffer_sec", 2.0)),
                conversation_log_dir=getattr(args, "conversation_log_dir", ""),
''',
    "engine kwargs")

assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] AH options/kwargs search removal done")
