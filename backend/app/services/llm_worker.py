"""LLM worker process: owns the ONNX model and runs generation requests one at a time.

``onnxruntime_genai`` holds the Python GIL inside its native calls. Prompt processing for a
7B model on CPU is a single call lasting a minute or more, which would freeze the API server's
event loop if it ran in the server process (a thread doesn't help). The model therefore lives
in this separate process.

Protocol: messages over a ``multiprocessing`` Pipe, each a ``(kind, request_id, payload)`` tuple.
    parent -> worker: ("generate", rid, {...}) | ("cancel", rid, None) | ("shutdown", None, None)
    worker -> parent: ("loaded", None, info) | ("load_error", None, message)
                      ("token", rid, text) | ("done", rid, stats) | ("error", rid, (code, message))
Requests are processed in order, and a cancel always arrives before the next request, so a
cancelled generation stops before the next one starts.

This module must stay importable without FastAPI or the rest of the app (it's imported by the
spawned process).
"""

from __future__ import annotations

from typing import Any

MIN_NEW_TOKENS = 64


def load_model(og: Any, model_path: str, execution_provider: str | None):
    import json
    import os

    config = og.Config(model_path)
    if execution_provider:
        config.clear_providers()
        if execution_provider.lower() != "cpu":
            config.append_provider(execution_provider)
    elif "-dml" in str(model_path).lower():
        try:
            config.clear_providers()
            config.append_provider("dml")
        except Exception:
            pass

    # Thread optimization: default to physical core count (6) when on CPU to prevent
    # excessive cache misses / context switches caused by logical hyperthreads.
    threads = os.environ.get("LEGAL_LLM_THREADS", "6")
    try:
        if not execution_provider or execution_provider.lower() == "cpu":
            config.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": int(threads)}}}}))
    except Exception:
        pass

    model = og.Model(config)
    return model, og.Tokenizer(model)


def _generate(conn, og, model, tokenizer, rid: int, request: dict, context_window: int) -> bool:
    """Run one request. Returns False if a shutdown was received mid-generation."""
    input_ids = tokenizer.encode(request["prompt"])
    prompt_tokens = len(input_ids)
    available = context_window - prompt_tokens
    if available < MIN_NEW_TOKENS:
        conn.send(("error", rid, ("PROMPT_TOO_LONG",
                                  f"Prompt uses {prompt_tokens} tokens; the context window is {context_window}.")))
        return True
    max_new = min(int(request["max_new_tokens"]), available)

    params = og.GeneratorParams(model)
    search = {"max_length": prompt_tokens + max_new, "repetition_penalty": request["repetition_penalty"]}
    if request["temperature"] > 0:
        search.update(do_sample=True, temperature=request["temperature"], top_p=request["top_p"])
    else:
        search["do_sample"] = False
    params.set_search_options(**search)
    generator = og.Generator(model, params)
    generator.append_tokens(input_ids)  # prompt processing: the long, GIL-holding call
    stream = tokenizer.create_stream()
    generated = 0
    keep_running = True
    while not generator.is_done():
        if conn.poll():
            kind, other, _ = conn.recv()
            if kind == "shutdown":
                keep_running = False
                break
            if kind == "cancel" and other == rid:
                break
            # A cancel for an already finished request: nothing to do.
        generator.generate_next_token()
        generated += 1
        piece = stream.decode(generator.get_next_tokens()[0])
        if piece:
            conn.send(("token", rid, piece))
    del generator
    conn.send(("done", rid, {"prompt_tokens": prompt_tokens, "generated_tokens": generated}))
    return keep_running


def worker_loop(conn, og: Any, model_path: str, execution_provider: str | None, context_window: int) -> None:
    try:
        model, tokenizer = load_model(og, model_path, execution_provider)
    except Exception as exc:
        conn.send(("load_error", None, f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("loaded", None, {"model_path": model_path}))

    while True:
        try:
            kind, rid, payload = conn.recv()
        except (EOFError, OSError):
            return  # parent went away
        if kind == "shutdown":
            return
        if kind != "generate":
            continue  # stale cancel
        try:
            if not _generate(conn, og, model, tokenizer, rid, payload, context_window):
                return
        except Exception as exc:
            conn.send(("error", rid, ("GENERATION_FAILED", f"{type(exc).__name__}: {exc}")))


def worker_main(conn, model_path: str, execution_provider: str | None, context_window: int) -> None:
    """Entry point of the spawned process."""
    try:
        import onnxruntime_genai as og
    except ImportError as exc:
        conn.send(("load_error", None, f"onnxruntime-genai is not installed: {exc}"))
        return
    worker_loop(conn, og, model_path, execution_provider, context_window)
