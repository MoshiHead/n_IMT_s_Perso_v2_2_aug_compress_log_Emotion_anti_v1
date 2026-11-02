"""fix3: install the two-thread (Moshi || render) producer in the AH server.
Reads the two new function bodies from scratch/new_producer_part{1,2}.txt."""
from pathlib import Path

ROOT = Path(r"e:\LAB\codeV9\Analysis\n_IMT_s_Perso_v2_2_aug_compress_log_Emotion_anti_v1")
P = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
part1 = (ROOT / "scratch" / "new_producer_part1.txt").read_text(encoding="utf-8-sig")
part2 = (ROOT / "scratch" / "new_producer_part2.txt").read_text(encoding="utf-8-sig")
src = P.read_text(encoding="utf-8")
orig = src


def region(text, start, end, new, desc):
    i = text.find(start)
    assert i >= 0, f"start not found: {desc}"
    j = text.find(end, i + len(start))
    assert j >= 0, f"end not found: {desc}"
    print(f"[{desc}] replaced {j - i} chars with {len(new)} chars")
    return text[:i] + new + text[j:]


def replace_once(text, old, new, desc):
    assert text.count(old) == 1, f"anchor not unique/found ({desc}): count={text.count(old)}"
    print(f"[edit] {desc}")
    return text.replace(old, new, 1)


# 1. Replace the whole single-thread producer with the two new functions.
src = region(
    src,
    "            def _gpu_producer_thread() -> None:",
    "            async def _get_media_epoch() -> float:",
    part1 + part2,
    "producer split",
)

# 2. Queue + thread declarations.
src = replace_once(
    src,
    '''        mic_q: queue.Queue[tuple[bytes, int] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None''',
    '''        mic_q: queue.Queue[tuple[bytes, int] | None] | None = None
        avatar_q: queue.Queue[dict | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        avatar_thread: threading.Thread | None = None''',
    "queue/thread declarations",
)

# 3. Queue creation: bounded avatar handoff queue (8 jobs ~= 7.7s of chunks).
src = replace_once(
    src,
    '''            mic_q = queue.Queue(maxsize=512)
            frame_q = asyncio.Queue(maxsize=512)''',
    '''            mic_q = queue.Queue(maxsize=512)
            avatar_q = queue.Queue(maxsize=8)
            frame_q = asyncio.Queue(maxsize=512)''',
    "queue creation",
)

# 4. Safe wrappers + thread starts.
NEW_WRAPPERS = '''            def _gpu_producer_thread_safe() -> None:
                """Defense-in-depth guard around the whole GPU producer loop.
                Without this, an uncaught exception ANYWHERE in that loop
                kills only this one daemon thread: audio/video generation
                stops forever, the websocket stays open, and neither
                conversation log records anything. This wrapper makes any
                future occurrence diagnosable (full traceback goes to the
                conversation log) and ends the session cleanly instead of
                hanging indefinitely."""
                try:
                    _gpu_producer_thread()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[GPU] FATAL uncaught exception in producer thread: {e!r}\\n{tb}",
                        flush=True,
                    )
                    with contextlib.suppress(Exception):
                        reply_engine.conv_logger.error("gpu_producer_thread", e, tb)
                    # Unblock the render thread so it can drain and exit; the
                    # direct sentinel puts below stop the senders regardless.
                    if avatar_q is not None:
                        with contextlib.suppress(Exception):
                            avatar_q.put_nowait(None)
                    for q in (frame_q, audio_q):
                        if q is None:
                            continue
                        with contextlib.suppress(Exception):
                            asyncio.run_coroutine_threadsafe(q.put(None), event_loop).result(timeout=30.0)

            def _avatar_render_thread_safe() -> None:
                """Same defense-in-depth guard for the avatar render thread:
                log any uncaught exception and stop the senders cleanly."""
                try:
                    _avatar_render_thread()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[GPU] FATAL uncaught exception in avatar render thread: {e!r}\\n{tb}",
                        flush=True,
                    )
                    with contextlib.suppress(Exception):
                        reply_engine.conv_logger.error("avatar_render_thread", e, tb)
                    for q in (frame_q, audio_q):
                        if q is None:
                            continue
                        with contextlib.suppress(Exception):
                            asyncio.run_coroutine_threadsafe(q.put(None), event_loop).result(timeout=30.0)

            gpu_thread = threading.Thread(target=_gpu_producer_thread_safe, daemon=True, name="gpu-producer")
            avatar_thread = threading.Thread(target=_avatar_render_thread_safe, daemon=True, name="avatar-render")
            gpu_thread.start()
            avatar_thread.start()
'''
src = region(
    src,
    "            def _gpu_producer_thread_safe() -> None:",
    '''            print(
                f"[AJ][AUDIO] session={session_id[:8]} "''',
    NEW_WRAPPERS,
    "safe wrappers + thread starts",
)

# 5. finally: join the render thread too (before dump_last_session).
src = replace_once(
    src,
    '''            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)''',
    '''            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)
            if avatar_thread is not None:
                await asyncio.to_thread(avatar_thread.join, 30.0)''',
    "finally: join avatar thread",
)

# 6. Clean stale _step_supports_text_token assignments + search comments.
src = replace_once(
    src,
    '''            # This override forwards text_token to prepare_step_input, and
            # process_transformer_output honors a provided token instead of
            # sampling one. That is what lets _step() force the model silent
            # while a search is in flight (see _step / suppress_text_until_ref).
            self._step_supports_text_token = True
            lm_gen.streaming_forever(1)''',
    '''            lm_gen.streaming_forever(1)''',
    "graphed path stale comment",
)
src = replace_once(
    src,
    '''        # step_with_layer takes only input_tokens, so text cannot be forced on
        # this fallback path; search-window suppression degrades to a no-op.
        self._step_supports_text_token = False
        lm_gen.streaming_forever(1)''',
    '''        lm_gen.streaming_forever(1)''',
    "fallback path stale comment",
)

assert src != orig
P.write_text(src, encoding="utf-8")
print("[ok] fix3 splice done")
