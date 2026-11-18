"""HTTP API. Chat and risk analysis stream Server-Sent Events over a POST response."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.container import Container
from app.errors import FileTooLargeError, LegalBotError
from app.schemas import ChatRequest, DocumentInfo, RiskAnalysisRequest, SummaryJob, SummaryRequest
from app.services.analysis_service import StreamEvent

logger = logging.getLogger(__name__)
router = APIRouter()

_UPLOAD_READ_SIZE = 1024 * 1024
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def get_container(request: Request) -> Container:
    return request.app.state.container


def format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


async def _sse(events: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield format_sse(event.event, event.data)
    except LegalBotError as exc:
        yield format_sse("error", exc.to_dict())
    except Exception:
        logger.exception("Stream failed")
        yield format_sse("error", {"code": "INTERNAL_ERROR", "message": "Unexpected error during analysis."})


# --- Health & Engine ------------------------------------------------------------------
@router.get("/health")
async def health(c: Container = Depends(get_container)) -> dict[str, Any]:
    return {
        "status": "ok",
        "llm": c.llm.status,
        "parsers_available": c.parser.available_strategies(),
        "word_converters": c.parser.available_office_backends(),
    }


@router.get("/engine")
async def get_engine(c: Container = Depends(get_container)) -> dict[str, str]:
    return {
        "engine": getattr(c.llm, "default_engine", "auto"),
        "provider": c.llm.status.get("provider", "unknown") or "unknown",
    }


@router.post("/engine")
async def set_engine(data: dict[str, str], c: Container = Depends(get_container)) -> dict[str, str]:
    engine = data.get("engine", "auto")
    c.llm.default_engine = engine
    return {"engine": engine, "status": "ok"}


# --- Documents ------------------------------------------------------------------------
@router.post("/documents", status_code=202, response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    password: str | None = Form(default=None),
    doc_type: str | None = Form(default=None),
    category: str | None = Form(default=None),
    c: Container = Depends(get_container),
) -> DocumentInfo:
    c.settings.tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = c.settings.tmp_dir / uuid.uuid4().hex
    try:
        size = 0
        with tmp_path.open("wb") as out:
            while chunk := await file.read(_UPLOAD_READ_SIZE):
                size += len(chunk)
                if size > c.settings.max_upload_bytes:
                    raise FileTooLargeError(f"The file exceeds the {c.settings.max_upload_mb} MB limit.")
                out.write(chunk)
        if size == 0:
            raise LegalBotError("The uploaded file is empty.")
        info = await c.ingestion.ingest(tmp_path, file.filename, password or None)
        if doc_type or category:
            info = c.registry.update(
                info.doc_id,
                doc_type=doc_type or "legal",
                category=category or "auto",
            )
        return info
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(c: Container = Depends(get_container)) -> list[DocumentInfo]:
    return c.registry.list()


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, c: Container = Depends(get_container)) -> DocumentInfo:
    return c.registry.get(doc_id)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, c: Container = Depends(get_container)) -> Response:
    c.registry.get(doc_id)
    c.summaries.cancel(doc_id)
    await c.vector_store.delete_document(doc_id)
    c.registry.delete(doc_id)
    return Response(status_code=204)


@router.get("/documents/{doc_id}/chunks")
async def list_chunks(doc_id: str, c: Container = Depends(get_container)) -> list[dict[str, Any]]:
    """Inspect how a document was chunked (useful when tuning parsing/chunking)."""
    c.registry.require_ready(doc_id)
    chunks = await c.vector_store.get_document_chunks(doc_id)
    return [{"chunk_id": ch.chunk_id, "text": ch.text, **ch.to_metadata()} for ch in chunks]


# --- Analysis (SSE) ----------------------------------------------------------------------
@router.post("/documents/{doc_id}/chat")
async def chat(doc_id: str, body: ChatRequest, c: Container = Depends(get_container)) -> StreamingResponse:
    c.registry.require_ready(doc_id)
    await c.llm.ensure_ready()  # fail with a proper HTTP status before the stream starts
    events = c.analysis.stream_answer(doc_id, body.question, body.top_k, engine=body.engine)
    return StreamingResponse(_sse(events), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/documents/{doc_id}/risk-analysis")
async def risk_analysis(
    doc_id: str, body: RiskAnalysisRequest, c: Container = Depends(get_container)
) -> StreamingResponse:
    doc = c.registry.require_ready(doc_id)
    await c.llm.ensure_ready()
    is_normal = (getattr(doc, "doc_type", "legal") == "normal")
    events = c.analysis.stream_risk_analysis(
        doc_id, body.party, body.focus, engine=body.engine, is_normal_doc=is_normal
    )
    return StreamingResponse(_sse(events), media_type="text/event-stream", headers=_SSE_HEADERS)


# --- Summary (background job) --------------------------------------------------------------
@router.post("/documents/{doc_id}/summary", status_code=202, response_model=SummaryJob)
async def start_summary(
    doc_id: str, body: SummaryRequest | None = None, c: Container = Depends(get_container)
) -> SummaryJob:
    c.registry.require_ready(doc_id)
    await c.llm.ensure_ready()
    return c.summaries.start(doc_id, force=bool(body and body.force))


@router.get("/documents/{doc_id}/summary", response_model=SummaryJob)
async def get_summary(doc_id: str, c: Container = Depends(get_container)) -> SummaryJob:
    c.registry.get(doc_id)
    job = c.summaries.get(doc_id)
    if job is None:
        raise _NoSummary()
    return job


class _NoSummary(LegalBotError):
    code = "SUMMARY_NOT_FOUND"
    status_code = 404

    def __init__(self) -> None:
        super().__init__("No summary has been generated for this document yet.")
