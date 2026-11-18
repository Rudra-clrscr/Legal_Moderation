"""RAG orchestration for Q&A and risk analysis.

Every stream emits, in order:
    sources    all excerpts sent to the LLM (exact text + section/page metadata)
    token      incremental generated text (many)
    citations  the excerpts the answer actually cited, validated against ``sources``
    risks      (risk analysis only) structured findings, each with validated citations
    done       disclaimer + flags
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

from app.config import Settings
from app.schemas import DISCLAIMER, SourceChunk
from app.services.citations import parse_risk_findings, resolve_citations
from app.services.llm_generator import ChatMessage, GenerationOptions, LLMGenerator
from app.services.prompts import (
    INSIGHTS_QUERIES,
    RISK_QUERIES,
    SYSTEM_PROMPT,
    build_insights_prompt,
    build_qa_prompt,
    build_risk_prompt,
    format_excerpt,
    format_excerpts,
)
from app.services.vector_store import RetrievedChunk, VectorStore

SAFETY_MARGIN_TOKENS = 64
NO_CONTEXT_MESSAGE = "The provided text does not contain passages relevant to this request."


@dataclass
class StreamEvent:
    event: str
    data: dict[str, Any]


def to_source(source_id: str, retrieved: RetrievedChunk) -> SourceChunk:
    c = retrieved.chunk
    return SourceChunk(
        source_id=source_id,
        chunk_id=c.chunk_id,
        section_label=c.section_label,
        section_title=c.section_title,
        header_path=c.header_path,
        page_start=c.page_start,
        page_end=c.page_end,
        text=c.text,
        score=None if retrieved.score is None else round(retrieved.score, 4),
    )


def interleave(result_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Round-robin merge of per-query rankings, de-duplicated, so every query is represented."""
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results) and results[rank].chunk.chunk_id not in seen:
                seen.add(results[rank].chunk.chunk_id)
                merged.append(results[rank])
    return merged


class AnalysisService:
    def __init__(self, vector_store: VectorStore, llm: LLMGenerator, settings: Settings) -> None:
        self._vs = vector_store
        self._llm = llm
        self._settings = settings

    async def stream_answer(
        self, doc_id: str, question: str, top_k: int | None = None, engine: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        retrieved = await self._vs.search(doc_id, question, top_k or self._settings.retrieval_top_k)
        async for event in self._stream_grounded(
            retrieved, lambda excerpts: build_qa_prompt(excerpts, question), parse_risks=False, engine=engine
        ):
            yield event

    async def stream_risk_analysis(
        self,
        doc_id: str,
        party: str | None = None,
        focus: str | None = None,
        engine: str | None = None,
        is_normal_doc: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        if is_normal_doc:
            queries = list(INSIGHTS_QUERIES)
            if focus and focus.strip():
                queries = [focus.strip(), f"{focus.strip()} findings methodology results"] + queries[:4]
            prompt_builder = lambda excerpts: build_insights_prompt(excerpts, focus)
        else:
            queries = list(RISK_QUERIES)
            if focus and focus.strip():
                queries = [focus.strip(), f"{focus.strip()} obligations liability restrictions"] + queries[:4]
            prompt_builder = lambda excerpts: build_risk_prompt(excerpts, party, focus)

        per_query = await self._vs.search_many(doc_id, queries, self._settings.risk_queries_top_k)
        async for event in self._stream_grounded(
            interleave(per_query),
            prompt_builder,
            parse_risks=True,
            engine=engine,
        ):
            yield event

    # -- Internals ----------------------------------------------------------------
    def _select_sources(self, retrieved: Sequence[RetrievedChunk], fixed_tokens: int) -> list[SourceChunk]:
        """Fit as many excerpts as the context window allows (in relevance order), then number
        them in document order so the model reads the contract sequentially."""
        budget = (
            self._settings.llm_context_window
            - self._settings.llm_max_new_tokens
            - fixed_tokens
            - SAFETY_MARGIN_TOKENS
        )
        chosen: list[RetrievedChunk] = []
        used = 0
        for item in retrieved:
            cost = self._llm.count_tokens(format_excerpt(to_source("S00", item))) + 2
            if used + cost > budget:
                continue
            chosen.append(item)
            used += cost
        chosen.sort(key=lambda r: r.chunk.index)
        return [to_source(f"S{i}", r) for i, r in enumerate(chosen, start=1)]

    async def _stream_grounded(
        self, retrieved, build_user, *, parse_risks: bool, engine: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        fixed_tokens = self._llm.count_tokens(
            self._llm.render_prompt(
                [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", build_user(""))]
            )
        )
        sources = self._select_sources(retrieved, fixed_tokens)
        yield StreamEvent("sources", {"sources": [s.model_dump() for s in sources]})

        if not sources:
            yield StreamEvent("token", {"text": NO_CONTEXT_MESSAGE})
            yield StreamEvent("citations", {"citations": [], "invalid_source_ids": []})
            if parse_risks:
                yield StreamEvent("risks", {"risks": []})
            yield StreamEvent("done", {"disclaimer": DISCLAIMER, "uncited": True})
            return

        messages = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", build_user(format_excerpts(sources))),
        ]
        options = GenerationOptions(
            max_new_tokens=self._settings.llm_max_new_tokens,
            temperature=self._settings.llm_temperature,
            top_p=self._settings.llm_top_p,
            repetition_penalty=self._settings.llm_repetition_penalty,
            engine=engine or "auto",
        )
        parts: list[str] = []
        async for piece in self._llm.stream(messages, options):
            parts.append(piece)
            yield StreamEvent("token", {"text": piece})

        answer = "".join(parts)
        by_id = {s.source_id: s for s in sources}
        cited, invalid = resolve_citations(answer, by_id.keys())
        yield StreamEvent(
            "citations",
            {"citations": [by_id[s].model_dump() for s in cited], "invalid_source_ids": invalid},
        )
        if parse_risks:
            findings = parse_risk_findings(answer, by_id.keys())
            yield StreamEvent("risks", {"risks": [f.model_dump() for f in findings]})
        yield StreamEvent("done", {"disclaimer": DISCLAIMER, "uncited": not cited})
