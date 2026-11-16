"""LLMGenerator <-> worker protocol tests.

The real ``worker_loop`` runs on a thread at the other end of a real Pipe, with a fake
``onnxruntime_genai`` module, so the streaming, cancellation and crash handling are exercised
end to end without a model. An optional test at the bottom runs the real 0.5B model, if it has
been converted, to check the event loop stays responsive during prompt processing.
"""

import asyncio
import multiprocessing
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.errors import LLMGenerationError, PromptTooLongError
from app.services.llm_generator import ChatMessage, GenerationOptions, LLMGenerator, render_chatml, render_mistral
from app.services.llm_worker import worker_loop

MESSAGES = [ChatMessage("system", "SYS"), ChatMessage("user", "Hello")]


def make_fake_og(tokens, step_delay=0.0, prompt_len=None):
    stats = {"steps": 0, "generators": 0}

    class Tokenizer:
        def __init__(self, model): ...

        def encode(self, text):
            return list(range(prompt_len if prompt_len is not None else len(text.split())))

        def create_stream(self):
            return SimpleNamespace(decode=lambda t: tokens[t])

    class GeneratorParams:
        def __init__(self, model):
            self.options = {}

        def set_search_options(self, **kw):
            self.options = kw

    class Generator:
        def __init__(self, model, params):
            self.i = -1
            stats["generators"] += 1
            stats["options"] = params.options

        def append_tokens(self, ids): ...

        def is_done(self):
            return self.i >= len(tokens) - 1

        def generate_next_token(self):
            time.sleep(step_delay)
            self.i += 1
            stats["steps"] += 1

        def get_next_tokens(self):
            return [self.i]

    class Config:
        def __init__(self, path): ...

    og = SimpleNamespace(Config=Config, Model=lambda cfg: object(), Tokenizer=Tokenizer,
                         GeneratorParams=GeneratorParams, Generator=Generator)
    return og, stats


def thread_worker(og, context_window=8192, crash_after_load=False):
    def factory(llm):
        parent, child = multiprocessing.Pipe(duplex=True)

        def run():
            if crash_after_load:
                child.send(("loaded", None, {}))
                child.recv()  # first request
                child.close()  # simulate the process dying mid-request
                return
            worker_loop(child, og, "fake", None, context_window)

        threading.Thread(target=run, daemon=True).start()
        return parent

    return factory


def make_llm(tmp_path, factory, context_window=8192) -> LLMGenerator:
    return LLMGenerator(tmp_path, context_window=context_window, worker_factory=factory)


def test_templates():
    assert render_chatml(MESSAGES) == (
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    )
    assert render_mistral(MESSAGES) == "[INST] SYS\n\nHello [/INST]"


def test_stream_yields_tokens_and_strips_special(tmp_path):
    og, stats = make_fake_og(["Based", " on", " the", " text", "<|im_end|>"])
    llm = make_llm(tmp_path, thread_worker(og))
    out = asyncio.run(llm.generate(MESSAGES, GenerationOptions(temperature=0.0)))
    assert out == "Based on the text"
    assert stats["options"]["do_sample"] is False and stats["options"]["max_length"] > 0
    assert llm.status["state"] == "ready"


def test_early_close_cancels_and_next_request_is_clean(tmp_path):
    og, stats = make_fake_og([f"t{i} " for i in range(5000)], step_delay=0.002)
    llm = make_llm(tmp_path, thread_worker(og))

    async def scenario():
        agen = llm.stream(MESSAGES)
        first = [await agen.__anext__() for _ in range(3)]
        await agen.aclose()  # e.g. client disconnected
        steps_at_cancel = stats["steps"]
        second = []
        async for piece in llm.stream(MESSAGES):  # must not receive leftovers of the first
            second.append(piece)
            if len(second) == 2:
                break
        return first, second, steps_at_cancel

    first, second, steps_at_cancel = asyncio.run(scenario())
    assert first == ["t0 ", "t1 ", "t2 "]
    assert second == ["t0 ", "t1 "]
    assert steps_at_cancel < 5000 and stats["generators"] == 2


def test_prompt_too_long(tmp_path):
    og, _ = make_fake_og(["x"], prompt_len=10_000)
    llm = make_llm(tmp_path, thread_worker(og, context_window=2048), context_window=2048)
    with pytest.raises(PromptTooLongError):
        asyncio.run(llm.generate([ChatMessage("user", "short")]))


def test_worker_crash_fails_request_and_reports_error(tmp_path):
    og, _ = make_fake_og(["x"])
    llm = make_llm(tmp_path, thread_worker(og, crash_after_load=True))
    with pytest.raises(LLMGenerationError):
        asyncio.run(llm.generate(MESSAGES))
    time.sleep(0.1)
    assert llm.status["state"] == "error"


def test_load_failure_is_reported(tmp_path):
    class BrokenConfig:
        def __init__(self, path):
            raise RuntimeError("bad model files")

    og, _ = make_fake_og(["x"])
    og.Config = BrokenConfig
    from app.errors import ModelUnavailableError

    llm = make_llm(tmp_path, thread_worker(og))
    with pytest.raises(ModelUnavailableError, match="bad model files"):
        llm.load()
    assert llm.status["state"] == "error"


def test_missing_model_reports_unavailable(tmp_path):
    from app.errors import ModelUnavailableError

    llm = LLMGenerator(tmp_path / "missing")
    with pytest.raises(ModelUnavailableError):
        llm.load()
    assert llm.status["state"] == "error"


SMALL_MODEL = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b-instruct-awq-onnx-int4"


@pytest.mark.skipif(not SMALL_MODEL.is_dir(), reason="converted 0.5B test model not present")
def test_real_worker_keeps_event_loop_responsive():
    """Regression test: in-process generation froze the event loop for the whole prefill."""
    llm = LLMGenerator(SMALL_MODEL, context_window=8192)
    long_prompt = [ChatMessage("user", "Summarise: " + "The Supplier shall deliver the goods on time. " * 450)]

    async def scenario():
        await llm.ensure_ready()
        assert llm.count_tokens(llm.render_prompt(long_prompt)) > 3000
        ticks, max_gap = 0, 0.0
        task = asyncio.create_task(llm.generate(long_prompt, GenerationOptions(max_new_tokens=8, temperature=0.0)))
        last = time.perf_counter()
        while not task.done():
            await asyncio.sleep(0.05)
            now = time.perf_counter()
            max_gap, last, ticks = max(max_gap, now - last), now, ticks + 1
        return await task, ticks, max_gap

    try:
        text, ticks, max_gap = asyncio.run(scenario())
    finally:
        llm.close()
    assert text and ticks > 20 and max_gap < 1.0
