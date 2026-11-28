"""Local LLM inference with ``onnxruntime-genai`` (quantised ONNX models), with async token streaming.

The model runs in a dedicated worker process (see ``llm_worker.py``): ``onnxruntime_genai``
holds the GIL during its native calls, so running it in the server process (even on a thread)
would freeze the event loop for the whole of prompt processing.

This class keeps the in-process API the services use (``ensure_ready``, ``count_tokens``,
``render_prompt``, ``stream``, ``generate``). A reader thread receives the worker's messages
and routes each token to the asyncio queue of the request it belongs to. Generations are
serialised (one model instance). If a consumer stops early (client disconnect), a cancel is
sent; the worker stops at its next token. If the worker dies, in-flight requests fail and the
next request starts a new worker.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import multiprocessing
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from app.config import Settings
from app.errors import LLMGenerationError, ModelUnavailableError, PromptTooLongError
from app.services.chunker import estimate_tokens
from app.branding import AI_MODEL_NAME
from app.services.llm_worker import MIN_NEW_TOKENS, worker_main

logger = logging.getLogger(__name__)

_SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "</s>")
# Keep the model loaded between uploads: a cold load adds ~5 s to the next request.
_OLLAMA_KEEP_ALIVE = "30m"
_LOAD_TIMEOUT_S = 900


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    engine: Literal["auto", "gpu", "cpu"] = "auto"
    json_mode: bool = False
    # Ollama only: constrain the reply to this JSON schema (implies json_mode). Other engines
    # fall back to plain JSON mode.
    json_schema: dict[str, Any] | None = field(default=None, hash=False, compare=False)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
def render_chatml(messages: Sequence[ChatMessage]) -> str:
    """Qwen2.5 chat format."""
    body = "".join(f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages)
    return body + "<|im_start|>assistant\n"


def render_mistral(messages: Sequence[ChatMessage]) -> str:
    """Mistral / SaulLM-7B-Instruct format. No system role: the system prompt is folded into the
    first user turn. BOS (``<s>``) is omitted because the tokenizer adds it."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    out: list[str] = []
    first_user = True
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            content = f"{system}\n\n{message.content}" if first_user and system else message.content
            out.append(f"[INST] {content} [/INST]")
            first_user = False
        else:
            out.append(f" {message.content}</s>")
    return "".join(out)


TEMPLATES = {"chatml": render_chatml, "mistral": render_mistral}

# Returns the parent end of a duplex Pipe connected to a running worker_loop.
WorkerFactory = Callable[["LLMGenerator"], Any]


def spawn_worker_process(llm: "LLMGenerator"):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=worker_main,
        args=(child, str(llm.model_path), llm.execution_provider, llm.context_window),
        name="llm-worker",
        daemon=True,
    )
    process.start()
    child.close()
    llm._process = process
    return parent


def _load_counting_tokenizer(model_path: Path):
    """Fast tokenizer for budgeting prompts without a round trip to the (busy) worker."""
    path = model_path / "tokenizer.json"
    if not path.is_file():
        return None
    try:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(path))
    except Exception:
        logger.warning("Could not load %s for token counting; using estimates", path, exc_info=True)
        return None


class LLMGenerator:
    def __init__(
        self,
        model_path: Path,
        *,
        provider: Literal["auto", "ollama", "onnx"] = "onnx",
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen2.5:3b",
        template: str = "chatml",
        execution_provider: str | None = None,
        context_window: int = 8192,
        defaults: GenerationOptions = GenerationOptions(),
        worker_factory: WorkerFactory = spawn_worker_process,
        display_name: str = AI_MODEL_NAME,
    ) -> None:
        if template not in TEMPLATES:
            raise ValueError(f"Unknown prompt template {template!r}; choose from {sorted(TEMPLATES)}")
        self.model_path = model_path
        self.provider = provider
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.execution_provider = execution_provider
        self.context_window = context_window
        self.defaults = defaults
        self.display_name = display_name
        self._render = TEMPLATES[template]
        self._worker_factory = worker_factory
        self._load_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._generation_lock: asyncio.Lock | None = None
        self._conn = None
        self._process = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._ids = itertools.count(1)
        self._tokenizer = None
        self._state: Literal["not_loaded", "loading", "ready", "error"] = "not_loaded"
        self._load_error: str | None = None
        self._active_provider: str = provider

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMGenerator":
        return cls(
            settings.llm_model_path,
            provider=settings.llm_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
            template=settings.llm_prompt_template,
            execution_provider=settings.llm_execution_provider,
            context_window=settings.llm_context_window,
            defaults=GenerationOptions(
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                repetition_penalty=settings.llm_repetition_penalty,
            ),
            display_name=settings.ai_model_name,
        )

    # -- Lifecycle -------------------------------------------------------------
    @property
    def status(self) -> dict[str, str | None]:
        # /health is public, so it reports the product name ("BONC Astra AI"), never the real model.
        # BONC Astra AI *is* the Qwen2.5 model: Ollama's ``ollama_model`` (qwen2.5:3b / qwen2.5:3b-cpu)
        # or the ONNX fallback in ``model_path``. The real one is logged when it loads ("LLM ready
        # using Ollama provider (...)" / "Starting LLM worker for ..."). "model_path" stays as a key
        # for older clients of /health.
        active = getattr(self, "_active_provider", self.provider)
        return {"state": self._state, "error": self._load_error, "provider": active,
                "model_name": self.display_name, "model_path": self.display_name}

    def _probe_ollama(self) -> bool:
        """Check if local Ollama server is running and responsive."""
        import httpx
        try:
            r = httpx.get(f"{self.ollama_base_url.rstrip('/')}/api/tags", timeout=1.5)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                logger.info("Ollama is reachable at %s (models: %s)", self.ollama_base_url, models)
                target_base = self.ollama_model.split(":")[0]
                has_model = any(self.ollama_model in m or target_base in m for m in models)
                if has_model or not models:
                    return True
                logger.warning("Configured Ollama model %r not in %s; will attempt to use anyway", self.ollama_model, models)
                return True
            return False
        except Exception as exc:
            logger.debug("Ollama probe check failed (%s): %s", self.ollama_base_url, exc)
            return False

    def load(self) -> None:
        """Blocking: start the worker or connect to Ollama. Safe to call repeatedly;
        retries after a failure or a worker crash."""
        with self._load_lock:
            if self._state == "ready":
                return
            self._state = "loading"

            # 1. Check Ollama if provider is "auto" or "ollama"
            if self.provider in ("auto", "ollama"):
                if self._probe_ollama():
                    self._active_provider = "ollama"
                    self._tokenizer = _load_counting_tokenizer(self.model_path)
                    self._state, self._load_error = "ready", None
                    logger.info("LLM ready using Ollama provider (%s)", self.ollama_model)
                    return
                elif self.provider == "ollama":
                    self._state, self._load_error = "error", f"Ollama not reachable at {self.ollama_base_url}"
                    raise ModelUnavailableError(self._load_error)
                else:
                    logger.info("Ollama not running; falling back to local ONNX model")

            # 2. Local ONNX worker process
            self._active_provider = "onnx"
            try:
                if not self.model_path.is_dir():
                    raise ModelUnavailableError(
                        f"ONNX model directory not found: {self.model_path}. See README for conversion steps."
                    )
                logger.info("Starting LLM worker for %s", self.model_path)
                conn = self._worker_factory(self)
                if not conn.poll(_LOAD_TIMEOUT_S):
                    raise ModelUnavailableError("Timed out waiting for the LLM worker to load the model.")
                kind, _, payload = conn.recv()
                if kind != "loaded":
                    conn.close()
                    raise ModelUnavailableError(f"Failed to load the LLM: {payload}")
                self._tokenizer = _load_counting_tokenizer(self.model_path)
                self._conn = conn
                self._reader = threading.Thread(target=self._read_loop, args=(conn,), name="llm-reader", daemon=True)
                self._reader.start()
                self._state, self._load_error = "ready", None
                logger.info("LLM ready via ONNX")
            except ModelUnavailableError as exc:
                self._state, self._load_error = "error", exc.message
                raise
            except (EOFError, OSError) as exc:
                self._state, self._load_error = "error", "The LLM worker exited while loading the model."
                raise ModelUnavailableError(self._load_error) from exc

    async def ensure_ready(self) -> None:
        if self._state != "ready":
            await asyncio.to_thread(self.load)

    def close(self) -> None:
        conn, process = self._conn, self._process
        self._state = "not_loaded"
        if conn is not None:
            try:
                self._send(("shutdown", None, None))
            except Exception:
                pass
        if process is not None:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()

    def _send(self, message: tuple) -> None:
        with self._send_lock:
            self._conn.send(message)

    def _read_loop(self, conn) -> None:
        while True:
            try:
                kind, rid, payload = conn.recv()
            except (EOFError, OSError):
                break
            target = self._pending.get(rid)
            if target is not None:
                loop, queue = target
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
                except RuntimeError:  # event loop closed
                    pass
        # The worker is gone (crash, OOM kill, shutdown): fail whatever was waiting.
        if self._conn is conn:
            if self._state == "ready":
                logger.error("LLM worker exited unexpectedly")
                self._state, self._load_error = "error", "The LLM worker process exited unexpectedly."
            self._conn = None
        for loop, queue in list(self._pending.values()):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("died", None))
            except RuntimeError:
                pass

    # -- Prompt helpers ------------------------------------------------------------
    def render_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return self._render(messages)

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return estimate_tokens(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    # -- Generation ------------------------------------------------------------------
    async def warm_up(self) -> None:
        """Load the model before the first request needs it (a cold load costs ~5 s).

        Only Ollama needs this: the ONNX worker loads in ``ensure_ready``. The options must
        match ``_stream_ollama``'s or the first real request reloads the model anyway.
        """
        await self.ensure_ready()
        if getattr(self, "_active_provider", None) != "ollama":
            return
        import httpx

        body = {"model": self.ollama_model, "prompt": "", "keep_alive": _OLLAMA_KEEP_ALIVE,
                "options": {"num_ctx": self.context_window}}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            await client.post(f"{self.ollama_base_url.rstrip('/')}/api/generate", json=body)

    async def _stream_ollama(
        self, messages: Sequence[ChatMessage], options: GenerationOptions
    ) -> AsyncIterator[str]:
        # Ollama's native /api/chat: its OpenAI-compatible /v1 endpoint ignores "options", which
        # left every request on the default 4096-token context (long resumes were cut off) and
        # reloaded the model whenever a native call used a different context.
        import httpx

        # Always the full window: a num_ctx that varies per request forces a model reload.
        ollama_opts: dict[str, Any] = {
            "num_ctx": self.context_window,
            "num_predict": options.max_new_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "repeat_penalty": options.repetition_penalty,
        }
        if options.engine == "cpu":
            ollama_opts["num_gpu"] = 0
        elif options.engine == "gpu":
            ollama_opts["num_gpu"] = 99
        body: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": ollama_opts,
            "keep_alive": _OLLAMA_KEEP_ALIVE,
        }
        if options.json_schema is not None:
            body["format"] = options.json_schema
        elif options.json_mode:
            body["format"] = "json"
        try:
            # Leaving this block (also when a consumer stops early) closes the HTTP stream, so
            # Ollama stops generating.
            async with httpx.AsyncClient(
                base_url=self.ollama_base_url.rstrip("/"), timeout=httpx.Timeout(120.0, connect=5.0)
            ) as client, client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise LLMGenerationError(f"Ollama returned HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMGenerationError(f"Ollama generation failed: {chunk['error']}")
                    piece = (chunk.get("message") or {}).get("content") or ""
                    for tok in _SPECIAL_TOKENS:
                        piece = piece.replace(tok, "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except LLMGenerationError:
            raise
        except Exception as exc:
            logger.exception("Ollama generation error: %s", exc)
            raise LLMGenerationError(f"Ollama generation failed: {exc}") from exc

    async def stream(
        self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None
    ) -> AsyncIterator[str]:
        await self.ensure_ready()
        opts = options or self.defaults

        if getattr(self, "_active_provider", None) == "ollama":
            async for piece in self._stream_ollama(messages, opts):
                yield piece
            return

        prompt = self.render_prompt(messages)
        if self.count_tokens(prompt) + MIN_NEW_TOKENS > self.context_window:
            raise PromptTooLongError(f"The prompt does not fit the {self.context_window}-token context window.")

        if self._generation_lock is None:
            self._generation_lock = asyncio.Lock()
        async with self._generation_lock:  # one model instance: serialise generations
            if self._state != "ready" or self._conn is None:
                await self.ensure_ready()
            rid = next(self._ids)
            queue: asyncio.Queue = asyncio.Queue()
            self._pending[rid] = (asyncio.get_running_loop(), queue)
            finished = False
            try:
                self._send(("generate", rid, {
                    "prompt": prompt,
                    "max_new_tokens": opts.max_new_tokens,
                    "temperature": opts.temperature,
                    "top_p": opts.top_p,
                    "repetition_penalty": opts.repetition_penalty,
                }))
                while True:
                    kind, payload = await queue.get()
                    if kind == "token":
                        text = str(payload)
                        for token in _SPECIAL_TOKENS:
                            text = text.replace(token, "")
                        if text:
                            yield text
                    elif kind == "done":
                        finished = True
                        return
                    elif kind == "error":
                        finished = True
                        code, message = payload
                        if code == "PROMPT_TOO_LONG":
                            raise PromptTooLongError(message)
                        raise LLMGenerationError(f"Generation failed: {message}")
                    elif kind == "died":
                        finished = True
                        raise LLMGenerationError("The LLM worker process exited during generation.")
            finally:
                self._pending.pop(rid, None)
                if not finished and self._conn is not None:
                    # Consumer stopped early. The cancel is ordered before any later request,
                    # so the worker stops this generation before starting the next one.
                    try:
                        self._send(("cancel", rid, None))
                    except (OSError, ValueError):
                        pass

    async def generate(self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None) -> str:
        return "".join([piece async for piece in self.stream(messages, options)])
