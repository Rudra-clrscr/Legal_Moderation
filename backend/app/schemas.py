"""Pydantic models shared by the API layer and the services."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

DISCLAIMER = (
    "This is an automated legal analysis based on the provided document text. "
    "It is not legal advice. Consult a qualified lawyer before relying on it."
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_pages(page_start: int, page_end: int) -> str:
    return f"p. {page_start}" if page_start == page_end else f"pp. {page_start}–{page_end}"


# --- Documents ----------------------------------------------------------------
class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ErrorInfo(BaseModel):
    code: str
    message: str


class ParseReport(BaseModel):
    parser_used: str
    page_count: int
    quality_score: float
    low_text_pages: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: DocumentStatus
    doc_type: str = "legal"  # "legal" | "normal"
    category: str = "auto"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    chunk_count: int = 0
    parse_report: ParseReport | None = None
    error: ErrorInfo | None = None


# --- Requests -------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    engine: Literal["auto", "gpu", "cpu"] | None = None


class RiskAnalysisRequest(BaseModel):
    party: str | None = Field(
        default=None,
        max_length=200,
        description="The party whose position is analysed, e.g. 'Customer' or 'Licensee'.",
    )
    focus: str | None = Field(
        default=None, max_length=500, description="Optional topic to focus on, e.g. 'termination'."
    )
    engine: Literal["auto", "gpu", "cpu"] | None = None


class SummaryRequest(BaseModel):
    force: bool = False
    engine: Literal["auto", "gpu", "cpu"] | None = None


# --- Citations & analysis results ---------------------------------------------------
class SourceChunk(BaseModel):
    """A document excerpt shown to the LLM, with everything the UI needs to display it."""

    source_id: str  # "S1", "S2", ... as referenced in the LLM output
    chunk_id: str
    section_label: str  # e.g. "Section 4.1"
    section_title: str  # e.g. "Termination for Convenience"
    header_path: list[str]
    page_start: int
    page_end: int
    text: str  # exact source text
    score: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> str:
        label = self.section_label
        if self.section_title and self.section_title.lower() not in label.lower():
            label = f"{label} — {self.section_title}"
        return f"Source: {label}, {format_pages(self.page_start, self.page_end)}"


class RiskFinding(BaseModel):
    title: str
    severity: Literal["High", "Medium", "Low", "Unspecified"]
    analysis: str
    source_ids: list[str] = Field(default_factory=list)
    invalid_source_ids: list[str] = Field(default_factory=list)
    citation_missing: bool = False


# --- Summaries -------------------------------------------------------------------
class SummaryStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionSummary(BaseModel):
    label: str
    page_start: int
    page_end: int
    summary: str


class SummaryJob(BaseModel):
    doc_id: str
    status: SummaryStatus
    progress_done: int = 0
    progress_total: int = 0
    summary: str | None = None
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    error: ErrorInfo | None = None
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    disclaimer: str = DISCLAIMER
