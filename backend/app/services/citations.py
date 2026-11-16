"""Citation extraction/validation and parsing of risk findings from LLM output.

The LLM cites excerpts as ``[S1]``. The backend resolves those identifiers against the excerpts
that were actually sent, so the UI only ever shows real source chunks, and any finding without a
valid citation is flagged rather than silently displayed as sourced.
"""

from __future__ import annotations

import re
from typing import Collection

from app.schemas import RiskFinding

_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")
_CITATION_BODY_RE = re.compile(
    r"^\s*(?:sources?\s*:?\s*)?S\d{1,3}(?:\s*(?:,|;|&|and)\s*S?\d{1,3})*\s*$", re.IGNORECASE
)
_SOURCE_ID_RE = re.compile(r"S?(\d{1,3})", re.IGNORECASE)


def extract_source_ids(text: str) -> list[str]:
    """``"... [S2] and [S1, S3] ... [Source: S2]"`` -> ``["S2", "S1", "S3"]`` (ordered, unique)."""
    found: list[str] = []
    for match in _BRACKET_RE.finditer(text):
        body = match.group(1)
        if not _CITATION_BODY_RE.match(body):
            continue
        body = re.sub(r"^\s*sources?\s*:?", "", body, flags=re.IGNORECASE)
        for number in _SOURCE_ID_RE.findall(body):
            sid = f"S{int(number)}"
            if sid not in found:
                found.append(sid)
    return found


def resolve_citations(text: str, valid_ids: Collection[str]) -> tuple[list[str], list[str]]:
    """Split cited identifiers into (valid, invalid/hallucinated)."""
    cited = extract_source_ids(text)
    return [s for s in cited if s in valid_ids], [s for s in cited if s not in valid_ids]


_RISK_HEADING_RE = re.compile(
    r"^\s*#{2,4}\s*Risk\s*\d+\s*[:.\-–—]?\s*(?P<title>.*)$", re.IGNORECASE | re.MULTILINE
)
_ANY_HEADING_RE = re.compile(r"^\s*#{1,4}\s", re.MULTILINE)
_SEVERITY_RE = re.compile(r"Severity[*\s:]*(High|Medium|Low)\b", re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"Analysis[*\s:]*(?P<body>.+)", re.IGNORECASE | re.DOTALL)
_FIELD_LINE_RE = re.compile(r"^\s*[*\-\s]*\**(Severity|Source|Sources)\**\s*:.*$", re.IGNORECASE | re.MULTILINE)


def parse_risk_findings(text: str, valid_ids: Collection[str]) -> list[RiskFinding]:
    headings = list(_RISK_HEADING_RE.finditer(text))
    findings: list[RiskFinding] = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        block = text[heading.end() : end]
        next_heading = _ANY_HEADING_RE.search(block)  # e.g. "### Overall Observations"
        if next_heading:
            block = block[: next_heading.start()]

        severity_match = _SEVERITY_RE.search(block)
        severity = severity_match.group(1).capitalize() if severity_match else "Unspecified"
        analysis_match = _ANALYSIS_RE.search(block)
        analysis = analysis_match.group("body") if analysis_match else _FIELD_LINE_RE.sub("", block)
        analysis = analysis.strip().strip("*").strip()

        # Prefer the claim-level citations inside the analysis: models sometimes put every
        # excerpt on the "Source:" line, which tells the reader nothing.
        valid, invalid = resolve_citations(analysis, valid_ids)
        if not valid:
            valid, invalid = resolve_citations(heading.group(0) + block, valid_ids)
        title = heading.group("title").strip().strip("*").strip() or f"Risk {i + 1}"
        findings.append(
            RiskFinding(
                title=title,
                severity=severity,  # type: ignore[arg-type]
                analysis=analysis,
                source_ids=valid,
                invalid_source_ids=invalid,
                citation_missing=not valid,
            )
        )
    return findings
