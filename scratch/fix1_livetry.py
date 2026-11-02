"""Edit liveTry.py: remove all online-search-only components (STT, query
router, context compressor, web search, ref LoRA, prompt settle). Keeps the
core Moshi engine, mic backlog cap, and conversation logging intact."""
from pathlib import Path

P = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1\IMTalker\liveTry.py")
src = P.read_text(encoding="utf-8")
orig = src


def cut_between(text, start_marker, end_marker, desc):
    i = text.find(start_marker)
    assert i >= 0, f"start marker not found ({desc})"
    j = text.find(end_marker, i + len(start_marker))
    assert j >= 0, f"end marker not found ({desc})"
    print(f"[cut] {desc}: removed {j - i} chars")
    return text[:i] + text[j:]


def replace_once(text, old, new, desc):
    assert text.count(old) == 1, f"anchor not unique/found ({desc}): count={text.count(old)}"
    print(f"[edit] {desc}")
    return text.replace(old, new, 1)


# 1. __init__ signature: drop all search-only params (keep backlog cap + log dir)
src = replace_once(
    src,
    '''        # --- STT + query routing + web search (all optional, off by default) ---
        ref_lora_dir: str = "",
        merge_ref_lora: bool = False,
        max_ref_tokens: int = 250,
        stt_hf_repo: str = "",
        stt_pkg_dir: str = "",
        vad_threshold: float = 0.5,
        suppress_text_during_search: bool = True,
        prompt_settle_sec: float = 0.0,
        stt_reject_foreign_script: bool = True,
        stt_max_non_latin_ratio: float = 0.15,
        stt_require_english: bool = True,
        max_input_buffer_sec: float = 2.0,
        compressor_model: str = "",
        compressor_device: str = "cuda",
        compressor_4bit: bool = True,
        compressor_max_passages: int = 2,
        router_threshold: float = 0.40,
        router_use_rules: bool = True,
        web_search_enabled: bool = False,
        web_search_api_key: str | None = None,
        web_search_provider: str = "tavily",
        web_search_max_results: int = 3,
        web_search_timeout: float = 3.0,
        web_search_min_score: float = 0.15,
        conversation_log_dir: str = "",
    ) -> None:''',
    '''        max_input_buffer_sec: float = 2.0,
        conversation_log_dir: str = "",
    ) -> None:''',
    "__init__ signature",
)

# 2. drop prompt_settle / suppress_text attribute init
src = replace_once(
    src,
    '''        # Read by _settle_after_prompt() / _start_thinking_sound(). These MUST be
        # set before _warmup_runtime() below, which calls reset_session() ->
        # _apply_system_prompts() -> _settle_after_prompt() during __init__.
        self.prompt_settle_sec = float(prompt_settle_sec)
        self.suppress_text_during_search = bool(suppress_text_during_search)
        self.step = 0''',
    '''        self.step = 0''',
    "prompt_settle attr init",
)

# 3. drop ref-LoRA loading block
src = replace_once(
    src,
    '''        # The reference LoRA must be applied here: after the base LM is loaded,
        # before LMGen(...)/CUDA-graph capture below (and before any subclass's
        # own graph-capture warmup) -- PEFT mutates self.lm's submodules in
        # place, so the graph bakes in the LoRA-augmented forward pass either
        # way. This adapter teaches the model to consume the injected
        # <lookup>/<ref> tags; it is unrelated to where the referenced text
        # came from, so it is still required now that the text comes from a web
        # search rather than a local document index.
        self.ref_lora_dir = str(ref_lora_dir or "")
        if self.ref_lora_dir:
            self._load_ref_lora(self.ref_lora_dir, merge_lora=bool(merge_ref_lora))

        self.mimi.eval()''',
    '''        self.mimi.eval()''',
    "ref lora load block",
)

# 4. drop the whole search state init + the four search loader methods
src = cut_between(
    src,
    "        # --- STT / query router / web search: all optional, each independently",
    "    def _resolve_voice_prompt_path(self) -> str:",
    "search state init + loader methods",
)

# 5. drop _settle_after_prompt method
src = cut_between(
    src,
    '''    @torch.no_grad()
    def _settle_after_prompt(self) -> None:''',
    "    def reset_session(self) -> None:",
    "_settle_after_prompt method",
)

# 6. drop its call site
src = replace_once(
    src,
    '''            self.lm_gen.step_system_prompts(self.mimi)
            self._settle_after_prompt()''',
    '''            self.lm_gen.step_system_prompts(self.mimi)''',
    "_settle_after_prompt call site",
)

# 7. reset_session: drop STT reset block
src = replace_once(
    src,
    '''        # Guarded with getattr: reset_session() is also called from inside
        # _warmup_runtime(), before the STT submodel (loaded at the very end
        # of __init__, after warmup) exists yet.
        stt_lm_gen = getattr(self, "stt_lm_gen", None)
        if stt_lm_gen is not None:
            with contextlib.suppress(Exception):
                stt_lm_gen.reset_streaming()
            with contextlib.suppress(Exception):
                self.stt_mimi.reset_streaming()
        self._apply_system_prompts()''',
    '''        self._apply_system_prompts()''',
    "reset_session STT reset",
)

assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] liveTry.py updated")

