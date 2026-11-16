"""Semantic chunking of per-page Markdown.

Chunks never straddle a heading boundary. Inside a section, paragraphs are packed up to a
target size, and a new chunk starts at each new numbered clause (``4.1``, ``4.2`` ...) once
the current chunk has substance, so citations point at precise clauses. Only paragraphs that
on their own exceed the size limit are split further (lines -> sentences -> word windows).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.services.document_parser import ParsedPage

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Cheap, tokenizer-independent estimate (~4 characters per token for English legal text)."""
    return max(1, math.ceil(len(text) / 4))


_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_EMPHASIS_RE = re.compile(r"[*_`]+")
_KEYWORD_NUM_RE = re.compile(
    r"^\s*(section|article|clause|schedule|exhibit|annex|appendix|part)\s+"
    r"([0-9]{1,3}(?:\.[0-9]{1,3})*|[IVXLC]+|[A-Z])\b\s*[.:)\-–—]*\s*",
    re.IGNORECASE,
)
_BARE_NUM_RE = re.compile(r"^\s*([0-9]{1,3}(?:\.[0-9]{1,3})*)[.)]?\s+(?=\S)")
# Multi-level clause numbers opening a paragraph: "4.1 ...", "**4.1.2** ...", "Section 4.1 ..."
_INLINE_CLAUSE_RE = re.compile(
    r"^\s*(?:[*_]{1,2}\s*)?(?:(?:section|clause)\s+)?([0-9]{1,3}(?:\.[0-9]{1,3})+)\.?(?=[\s*_):]|$)",
    re.IGNORECASE,
)
_FLAT_CLAUSE_RE = re.compile(r"^\s*(?:[*_]{1,2}\s*)?([0-9]{1,2})\.(?:[*_]{1,2})?\s+(?=[A-Z(\"“‘'])")
_NUMBER_ONLY_RE = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3})*\.?\s*[-–—]?\s*")
_LINE_SPLIT_RE = re.compile(r"\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"“0-9])")


def clean_title(title: str) -> str:
    return _EMPHASIS_RE.sub("", title).strip()


def split_numbered_title(title: str) -> tuple[str | None, str]:
    """``"4.1 Termination"`` -> ``("Section 4.1", "Termination")``; unnumbered -> ``(None, title)``."""
    match = _KEYWORD_NUM_RE.match(title)
    if match:
        kind, number = match.group(1).capitalize(), match.group(2)
        number = number.upper() if not number[0].isdigit() else number
        return f"{kind} {number}", title[match.end():].strip() or title
    match = _BARE_NUM_RE.match(title)
    if match:
        return f"Section {match.group(1)}", title[match.end():].strip()
    return None, title


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str  # exact source text, shown to the user as the citation body
    header_path: list[str]
    section_label: str
    section_title: str
    page_start: int
    page_end: int
    token_estimate: int

    @property
    def embedding_text(self) -> str:
        """Header context improves retrieval for short clauses ("30 days" means little alone)."""
        if not self.header_path:
            return self.text
        return f"{' > '.join(self.header_path)}\n\n{self.text}"

    def to_metadata(self) -> dict[str, Any]:
        """Chroma-safe metadata: only str/int/float/bool, no None or lists."""
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.index,
            "header_path": " > ".join(self.header_path),
            "section_label": self.section_label,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_record(cls, chunk_id: str, text: str, meta: Mapping[str, Any]) -> "Chunk":
        header_path = str(meta.get("header_path") or "")
        return cls(
            chunk_id=chunk_id,
            doc_id=str(meta["doc_id"]),
            index=int(meta.get("chunk_index", 0)),
            text=text,
            header_path=[h for h in header_path.split(" > ") if h],
            section_label=str(meta.get("section_label") or ""),
            section_title=str(meta.get("section_title") or ""),
            page_start=int(meta.get("page_start", 1)),
            page_end=int(meta.get("page_end", 1)),
            token_estimate=int(meta.get("token_estimate", 0)),
        )


@dataclass(frozen=True)
class _Block:
    text: str
    page: int
    header_path: tuple[str, ...]
    clause: str | None  # detected or inherited clause number within the section


@dataclass
class _Draft:
    blocks: list[_Block] = field(default_factory=list)
    tokens: int = 0

    def clauses(self) -> list[str]:
        seen: list[str] = []
        for block in self.blocks:
            if block.clause and block.clause not in seen:
                seen.append(block.clause)
        return seen


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class MarkdownChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 350,
        max_tokens: int = 480,
        min_tokens: int = 60,
        token_counter: TokenCounter = estimate_tokens,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Require 0 < min_tokens <= target_tokens <= max_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.count = token_counter

    def chunk(self, doc_id: str, pages: Sequence[ParsedPage]) -> list[Chunk]:
        drafts: list[_Draft] = []
        current = _Draft()

        for block in self._split_oversized(_iter_blocks(pages)):
            tokens = self.count(block.text)
            if current.blocks:
                new_section = block.header_path != current.blocks[0].header_path
                new_clause = (
                    block.clause is not None
                    and block.clause not in current.clauses()
                    and current.tokens >= self.min_tokens
                )
                full = current.tokens >= self.target_tokens or current.tokens + tokens > self.max_tokens
                if new_section or new_clause or full:
                    drafts.append(current)
                    current = _Draft()
            current.blocks.append(block)
            current.tokens += tokens
        if current.blocks:
            drafts.append(current)

        return [self._finalize(doc_id, i, d) for i, d in enumerate(drafts)]

    def _finalize(self, doc_id: str, index: int, draft: _Draft) -> Chunk:
        header_path = list(draft.blocks[0].header_path)
        clauses = draft.clauses()
        header_label, title = None, ""
        for header in reversed(header_path):
            header_label, title = split_numbered_title(header)
            if header_label:
                break
        if not header_label and header_path:
            title = header_path[-1]

        if len(clauses) == 1:
            label = f"Section {clauses[0]}"
        elif clauses:
            label = f"Sections {clauses[0]}–{clauses[-1]}"
        else:
            label = header_label or (header_path[-1] if header_path else "Preamble")

        text = "\n\n".join(b.text for b in draft.blocks)
        pages = [b.page for b in draft.blocks]
        return Chunk(
            chunk_id=f"{doc_id}:{index:05d}",
            doc_id=doc_id,
            index=index,
            text=text,
            header_path=header_path,
            section_label=label,
            section_title=title if title != label else "",
            page_start=min(pages),
            page_end=max(pages),
            token_estimate=self.count(text),
        )

    # -- Oversized paragraph handling ------------------------------------------------
    def _split_oversized(self, blocks: Iterable[_Block]) -> Iterable[_Block]:
        for block in blocks:
            if self.count(block.text) <= self.max_tokens:
                yield block
                continue
            for part in self._split_text(block.text):
                yield _Block(part, block.page, block.header_path, block.clause)

    def _split_text(self, text: str) -> list[str]:
        if self.count(text) <= self.max_tokens:
            return [text]
        for splitter, joiner in ((_LINE_SPLIT_RE, "\n"), (_SENTENCE_SPLIT_RE, " ")):
            units = [u.strip() for u in splitter.split(text) if u.strip()]
            if len(units) > 1:
                parts: list[str] = []
                for unit in units:
                    parts.extend(self._split_text(unit))
                return self._pack(parts, joiner)
        words = text.split()
        size = max(1, len(words) * self.target_tokens // max(1, self.count(text)))
        return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    def _pack(self, parts: Sequence[str], joiner: str) -> list[str]:
        packed: list[str] = []
        buffer: list[str] = []
        used = 0
        for part in parts:
            cost = self.count(part)
            if buffer and used + cost > self.target_tokens:
                packed.append(joiner.join(buffer))
                buffer, used = [], 0
            buffer.append(part)
            used += cost
        if buffer:
            packed.append(joiner.join(buffer))
        return packed


def _clause_match(text: str, stack: Sequence[tuple[int, str]]) -> re.Match | None:
    match = _INLINE_CLAUSE_RE.match(text)
    if not match and not any(split_numbered_title(t)[0] for _, t in stack):
        # Flat "1. ... 20." clause numbering is only trusted when no enclosing heading is
        # numbered; under "12. INDEMNITY" a "1." is a list item, not Section 1.
        match = _FLAT_CLAUSE_RE.match(text)
    return match


def _starts_clause(line: str, stack: Sequence[tuple[int, str]]) -> bool:
    return _clause_match(line.lstrip("-*+ \t"), stack) is not None


def _iter_blocks(pages: Sequence[ParsedPage]) -> list[_Block]:
    """Split pages into paragraph blocks annotated with page, heading path and clause number."""
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    clause: str | None = None

    def flush(lines: list[str], page: int) -> None:
        nonlocal clause
        text = "\n".join(lines).strip()
        if not text:
            return
        match = _clause_match(text.lstrip("-*+ \t"), stack)
        if match:
            clause = match.group(1)
        blocks.append(_Block(text, page, tuple(t for _, t in stack), clause))

    pending_number: str | None = None  # "## 12." followed by "## INDEMNITY" -> "12. INDEMNITY"

    for page in pages:
        lines: list[str] = []
        for line in page.markdown.splitlines():
            header = _HEADER_RE.match(line)
            if header or not line.strip():
                flush(lines, page.number)
                lines = []
                if header:
                    level, title = len(header.group(1)), clean_title(header.group(2))
                    if _NUMBER_ONLY_RE.fullmatch(title):
                        pending_number = title.rstrip(" .-–—")
                        continue
                    if pending_number and title:
                        title = f"{pending_number}. {title}"
                    pending_number = None
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    if title:
                        stack.append((level, title))
                    clause = None  # clause numbering is scoped to its heading
                continue
            if pending_number:  # the number heading belonged to body text, not a heading
                lines.append(pending_number)
                pending_number = None
            if lines and _starts_clause(line, stack):  # list-formatted clauses without blank lines
                flush(lines, page.number)
                lines = []
            lines.append(line.rstrip())
        flush(lines, page.number)
    return blocks
