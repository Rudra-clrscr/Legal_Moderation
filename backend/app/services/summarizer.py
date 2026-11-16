"""Map-reduce summarisation for documents longer than the LLM context window.

Map:    consecutive chunks are packed into batches (preferring top-level section boundaries)
        and each batch is summarised on its own.
Reduce: the batch summaries are combined into one executive summary. If they do not fit the
        context window, they are first condensed in groups (recursively, bounded depth).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Sequence

from app.config import Settings
from app.errors import DocumentUnreadableError, LegalBotError
from app.schemas import ErrorInfo, SectionSummary, SummaryJob, SummaryStatus, format_pages, utcnow
from app.services.chunker import Chunk
from app.services.document_registry import DocumentRegistry
from app.services.llm_generator import ChatMessage, GenerationOptions, LLMGenerator
from app.services.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    build_collapse_prompt,
    build_map_prompt,
    build_reduce_prompt,
)
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_COLLAPSE_DEPTH = 3
PROMPT_OVERHEAD_TOKENS = 400
SUMMARY_FILE = "summary.json"

ProgressCallback = Callable[[int, int], None]


def _format_chunk(chunk: Chunk) -> str:
    heading = chunk.section_label + (f" — {chunk.section_title}" if chunk.section_title else "")
    return f"[{heading}, {format_pages(chunk.page_start, chunk.page_end)}]\n{chunk.text}"


def _batch_label(batch: Sequence[Chunk]) -> str:
    first, last = batch[0].section_label, batch[-1].section_label
    return first if first == last else f"{first} – {last}"


class DocumentSummarizer:
    def __init__(self, vector_store: VectorStore, llm: LLMGenerator, settings: Settings) -> None:
        self._vs = vector_store
        self._llm = llm
        self._settings = settings

    def _options(self, max_new_tokens: int) -> GenerationOptions:
        return GenerationOptions(
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # deterministic summaries
            top_p=1.0,
            repetition_penalty=self._settings.llm_repetition_penalty,
        )

    async def _ask(self, user_prompt: str, max_new_tokens: int) -> str:
        messages = [ChatMessage("system", SUMMARY_SYSTEM_PROMPT), ChatMessage("user", user_prompt)]
        return (await self._llm.generate(messages, self._options(max_new_tokens))).strip()

    def make_batches(self, chunks: Sequence[Chunk]) -> list[list[Chunk]]:
        budget = self._settings.summary_map_input_tokens
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        used = 0
        current_top: str | None = None
        for chunk in chunks:
            cost = self._llm.count_tokens(_format_chunk(chunk))
            top = chunk.header_path[0] if chunk.header_path else ""
            boundary = current_top is not None and top != current_top and used >= budget * 0.5
            if current and (used + cost > budget or boundary):
                batches.append(current)
                current, used = [], 0
            current.append(chunk)
            used += cost
            current_top = top
        if current:
            batches.append(current)
        return batches

    async def summarize(
        self, doc_id: str, on_progress: ProgressCallback | None = None
    ) -> tuple[str, list[SectionSummary]]:
        await self._llm.ensure_ready()
        chunks = await self._vs.get_document_chunks(doc_id)
        if not chunks:
            raise DocumentUnreadableError("The document has no indexed text to summarise.")

        batches = self.make_batches(chunks)
        total = len(batches) + 1
        section_summaries: list[SectionSummary] = []
        for i, batch in enumerate(batches, start=1):
            text = "\n\n".join(_format_chunk(c) for c in batch)
            summary = await self._ask(build_map_prompt(text), self._settings.summary_map_max_new_tokens)
            section_summaries.append(
                SectionSummary(
                    label=_batch_label(batch),
                    page_start=min(c.page_start for c in batch),
                    page_end=max(c.page_end for c in batch),
                    summary=summary,
                )
            )
            if on_progress:
                on_progress(i, total)

        # Neutral part headers: a section-like batch label ("Sections 16-17") gets copied by the
        # model as a citation for everything in the batch.
        parts = [
            f"--- Part {i} ({format_pages(s.page_start, s.page_end)}) ---\n{s.summary}"
            for i, s in enumerate(section_summaries, start=1)
        ]
        final = await self._reduce(parts)
        if on_progress:
            on_progress(total, total)
        return final, section_summaries

    async def _reduce(self, parts: list[str], depth: int = 0) -> str:
        budget = (
            self._settings.llm_context_window
            - self._settings.summary_reduce_max_new_tokens
            - PROMPT_OVERHEAD_TOKENS
        )
        joined = "\n\n".join(parts)
        if len(parts) == 1 or depth >= MAX_COLLAPSE_DEPTH or self._llm.count_tokens(joined) <= budget:
            return await self._ask(build_reduce_prompt(joined), self._settings.summary_reduce_max_new_tokens)

        groups: list[list[str]] = []
        current: list[str] = []
        used = 0
        for part in parts:
            cost = self._llm.count_tokens(part)
            if current and used + cost > budget:
                groups.append(current)
                current, used = [], 0
            current.append(part)
            used += cost
        if current:
            groups.append(current)

        condensed = [
            await self._ask(build_collapse_prompt("\n\n".join(g)), self._settings.summary_map_max_new_tokens)
            for g in groups
        ]
        return await self._reduce(condensed, depth + 1)


class SummaryJobManager:
    """Runs summaries as background tasks (they can take minutes on CPU) and persists results."""

    def __init__(self, summarizer: DocumentSummarizer, registry: DocumentRegistry) -> None:
        self._summarizer = summarizer
        self._registry = registry
        self._jobs: dict[str, SummaryJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def get(self, doc_id: str) -> SummaryJob | None:
        if doc_id in self._jobs:
            return self._jobs[doc_id]
        stored = self._registry.read_artifact(doc_id, SUMMARY_FILE)
        if stored:
            job = SummaryJob.model_validate_json(stored)
            self._jobs[doc_id] = job
            return job
        return None

    def start(self, doc_id: str, force: bool = False) -> SummaryJob:
        existing = self.get(doc_id)
        if existing and existing.status == SummaryStatus.RUNNING:
            return existing
        if existing and existing.status == SummaryStatus.COMPLETED and not force:
            return existing
        job = SummaryJob(doc_id=doc_id, status=SummaryStatus.RUNNING)
        self._jobs[doc_id] = job
        self._tasks[doc_id] = asyncio.create_task(self._run(job))
        return job

    def cancel(self, doc_id: str) -> None:
        task = self._tasks.pop(doc_id, None)
        if task and not task.done():
            task.cancel()
        self._jobs.pop(doc_id, None)

    async def _run(self, job: SummaryJob) -> None:
        def progress(done: int, total: int) -> None:
            job.progress_done, job.progress_total = done, total

        try:
            summary, sections = await self._summarizer.summarize(job.doc_id, progress)
            job.summary, job.section_summaries = summary, sections
            job.status = SummaryStatus.COMPLETED
        except asyncio.CancelledError:
            raise
        except LegalBotError as exc:
            job.status, job.error = SummaryStatus.FAILED, ErrorInfo(code=exc.code, message=exc.message)
        except Exception:
            logger.exception("Summary failed for %s", job.doc_id)
            job.status = SummaryStatus.FAILED
            job.error = ErrorInfo(code="INTERNAL_ERROR", message="Unexpected error while summarising.")
        finally:
            self._tasks.pop(job.doc_id, None)
        job.completed_at = utcnow()
        if job.status == SummaryStatus.COMPLETED:
            try:
                self._registry.write_artifact(job.doc_id, SUMMARY_FILE, job.model_dump_json(indent=2))
            except LegalBotError:
                pass  # document deleted meanwhile
