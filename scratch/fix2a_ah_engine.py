"""AH server search removal, part A: MoshiOnlyEngineWithHidden class edits.
Removes thinking sound, STT/router/search/consume machinery, force_idle, and
the search-latency bookkeeping from _step."""
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


# A1. __init__ signature: drop thinking_sound_path / search_max_filler_sec
src = region(src,
    "        capture_layer: int = -2,",
    "        **kwargs,",
    "        capture_layer: int = -2,\n",
    "init signature")

# A2. __init__ body: filler-cap + thinking-sound loading
src = region(src,
    "        # Forensic finding (conversation_logs_1/2/3): the old fixed 25-chunk",
    "        self._install_graph_hidden_capture()",
    "",
    "init body")

# B. thinking-sound methods (_next_thinking_sound_chunk .. _stop_thinking_sound)
src = region(src,
    "    def _next_thinking_sound_chunk(self) -> np.ndarray:",
    "    def _install_graph_hidden_capture(self) -> None:",
    "",
    "thinking sound methods")

# C. turn-detection class constants (VAD / filler cap / speech RMS / text-token flag)
src = region(src,
    "    # -- Turn-detection constants (frame = one MIMI_FRAME_SIZE / 80ms chunk,",
    "    def reset_session(self) -> None:",
    "",
    "turn-detection constants")

# D. reset_session -> minimal
src = region(src,
    """    def reset_session(self) -> None:
        # A session that ends (or is restarted by a reconnect) with a turn still""",
    "    def _inject_tokens(self, tokens: list[int]) -> None:",
    '''    def reset_session(self) -> None:
        # Search/turn-detection machinery removed: a session reset now only
        # resets the core Moshi conversation state.
        super().reset_session()

''',
    "reset_session")

# E. _inject_tokens .. _consume_pending block (ends right before _step)
src = region(src,
    "    def _inject_tokens(self, tokens: list[int]) -> None:",
    """    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:""",
    "",
    "inject/stt/route/search/consume block")

# F. _step: STT/consume hooks
src = region(src,
    "        # -- STT/VAD forward pass + turn-boundary detection, then consume any",
    "        t_lm0 = time.perf_counter()",
    "",
    "step stt/consume hooks")

# G. _step: suppress-text branch -> plain lm step
src = region(src,
    """        if self.suppress_text_until_ref and self._step_supports_text_token:
            lm_out = self.lm_gen._step(
                codes[:, :, :1], text_token=getattr(self.lm_gen, "zero_text_code", 3)
            )
        else:
            lm_out = self.lm_gen._step(codes[:, :, :1])""",
    "        t_lm1 = time.perf_counter()",
    "        lm_out = self.lm_gen._step(codes[:, :, :1])\n",
    "suppress branch")

# H. _step: last-text-emit bookkeeping
src = region(src,
    """            if token_piece:
                self.audio_text += token_piece""",
    "            reply_codes = tokens[:, 1:].detach()",
    """            if token_piece:
                self.audio_text += token_piece
""",
    "token piece block")

# I. _step: model_own_rms + first-speech latency + thinking-sound swap
src = region(src,
    "        # The model's OWN output level, measured before the thinking sound can",
    "        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))",
    "",
    "model_own_rms/first-speech/thinking swap")

# J. _step: per-frame latency accumulation
src = region(src,
    "        # Charge this 80ms frame's GPU cost to the turn being timed. Accumulated",
    "        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)",
    "",
    "latency accumulation")

# K. _step: drop force_idle from the returned event
old_k = '            "force_idle": force_idle,'
assert src.count(old_k) == 1, f"force_idle line count={src.count(old_k)}"
src = src.replace(old_k + "\n", "")
print("[return dict] force_idle removed")

assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] AH engine-class search removal done")
