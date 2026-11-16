"""PDF / image -> per-page Markdown.

This module knows nothing about embeddings or LLMs. It turns an uploaded file into
``ParsedDocument`` (a list of Markdown pages) using a chain of extraction strategies:

    1. marker               Marker (Surya OCR + layout) - best structure, handles scans
    2. marker_enhanced_ocr  re-render pages, clean them up (contrast/denoise/sharpen), force OCR
    3. unstructured         Unstructured.io hi_res partitioning
    4. pypdf                plain text layer + heuristic headings (always available, no OCR)

Every attempt gets a quality score. The first attempt that reaches the configured threshold
wins; otherwise the best attempt is kept and flagged with warnings. Attempts below the
minimum quality raise ``DocumentUnreadableError``.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from app.config import Settings
from app.errors import (
    DocumentUnreadableError,
    InvalidPasswordError,
    LegalBotError,
    PasswordRequiredError,
    TooManyPagesError,
    UnsupportedFileTypeError,
)
from app.schemas import ParseReport
from app.services.office_converter import OfficeConverter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ParsedPage:
    number: int  # 1-based
    markdown: str


@dataclass
class PreparedDocument:
    """A PDF ready for extraction (decrypted / converted from an image if needed)."""

    path: Path
    is_temporary: bool  # True when ``path`` is a derived file that must be deleted after parsing
    source_type: str  # "pdf" | "image"
    page_count: int | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]
    parser_used: str
    quality_score: float
    low_text_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return "\n\n".join(f"<!-- page {p.number} -->\n\n{p.markdown}" for p in self.pages)

    def report(self) -> ParseReport:
        return ParseReport(
            parser_used=self.parser_used,
            page_count=len(self.pages),
            quality_score=round(self.quality_score, 3),
            low_text_pages=self.low_text_pages,
            warnings=self.warnings,
        )


class ExtractionStrategy(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def extract(self, pdf_path: Path) -> list[ParsedPage]: ...


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------
_MD_NOISE_RE = re.compile(r"<[^>]+>|[#*_`|>]|-{3,}|={3,}")
# Tokens that look like real words, numbers, clause refs or list markers (vs. OCR garbage).
_PLAUSIBLE_TOKEN_RE = re.compile(
    r"[(\[\"'“‘]*"
    r"(?:[A-Za-z][a-z'’\-]*|[A-Z][A-Z'’\-]+|\$?\d+(?:[.,:/\-]\d+)*%?|\(?[a-zA-Z0-9]{1,4}\)|[ivxlcIVXLC]+)"
    r"[)\]\"'”’.,;:!?%]*"
)


@dataclass
class QualityAssessment:
    score: float
    low_text_pages: list[int]


def assess_quality(pages: Sequence[ParsedPage], min_chars_per_page: int) -> QualityAssessment:
    """Score 0..1 = (share of plausible tokens) x (text density relative to ``min_chars_per_page``)."""
    if not pages:
        return QualityAssessment(0.0, [])
    total_tokens = plausible = total_letters = 0
    low_text_pages: list[int] = []
    for page in pages:
        text = _MD_NOISE_RE.sub(" ", page.markdown)
        tokens = text.split()
        letters = sum(ch.isalpha() for ch in text)
        total_tokens += len(tokens)
        plausible += sum(1 for t in tokens if _PLAUSIBLE_TOKEN_RE.fullmatch(t))
        total_letters += letters
        if letters < min_chars_per_page * 0.25:
            low_text_pages.append(page.number)
    if total_tokens == 0:
        return QualityAssessment(0.0, low_text_pages)
    word_ratio = plausible / total_tokens
    density = min(1.0, (total_letters / len(pages)) / max(1, min_chars_per_page))
    return QualityAssessment(word_ratio * density, low_text_pages)


_SPAN_ANCHOR_RE = re.compile(r"<span id=\"[^\"]*\"></span>")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_EMPTY_CELLS_RE = re.compile(r"(?:\|\s*){2,}\|")
_TABLE_RULE_RE = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*$", re.MULTILINE)
# OCR'd scans often yield trivial LaTeX such as "$-$" or "$\cdot$" for dashes and bullets.
_TRIVIAL_MATH_RE = re.compile(r"\$\s*(\\?[^\s$A-Za-z0-9]{0,6}|\\[a-z]{2,8})\s*\$")


def clean_markdown(md: str) -> str:
    md = _SPAN_ANCHOR_RE.sub("", md)
    md = _IMAGE_RE.sub("", md)
    md = _TRIVIAL_MATH_RE.sub(" ", md)
    # Layout models sometimes read scanned text as wide tables padded with hundreds of spaces;
    # the padding wastes embedding and LLM context.
    md = _TABLE_RULE_RE.sub("|---|", md)
    md = _EMPTY_CELLS_RE.sub("|", md)
    md = _SPACE_RUN_RE.sub(" ", md)
    md = md.replace("\r\n", "\n").replace("\x00", "")
    return _BLANK_RUN_RE.sub("\n\n", md).strip()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
_MARKER_PAGE_RE = re.compile(r"\{(\d+)\}-{48}")


def split_marker_pages(text: str) -> list[ParsedPage]:
    """Split Marker's ``paginate_output`` Markdown (``{0}------...``) into pages."""
    parts = _MARKER_PAGE_RE.split(text)
    if len(parts) < 3:
        return [ParsedPage(1, text)]
    pages = [ParsedPage(int(pid) + 1, content) for pid, content in zip(parts[1::2], parts[2::2])]
    if parts[0].strip():
        pages[0].markdown = parts[0] + "\n\n" + pages[0].markdown
    return pages


class MarkerStrategy:
    """Marker (by the Surya team): layout-aware PDF -> Markdown with built-in OCR."""

    name = "marker"
    _models: dict | None = None
    _models_lock = threading.Lock()

    def __init__(self, force_ocr: bool = False) -> None:
        self.force_ocr = force_ocr

    def is_available(self) -> bool:
        return _has_module("marker")

    @classmethod
    def _artifacts(cls) -> dict:
        with cls._models_lock:
            if cls._models is None:
                from marker.models import create_model_dict

                logger.info("Loading Marker/Surya models (first use)...")
                cls._models = create_model_dict()
            return cls._models

    def extract(self, pdf_path: Path) -> list[ParsedPage]:
        if sys.platform == "win32" and not sys.flags.utf8_mode:
            # indra_ocr (Marker's OCR engine) reads its UTF-8 model metadata with the locale
            # codec, which fails on Windows (cp1252). Python's UTF-8 mode fixes it.
            raise RuntimeError(
                "Marker needs Python UTF-8 mode on Windows. Start the server with "
                "'python run_server.py' or 'python -X utf8 -m uvicorn app.main:app'."
            )
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered

        config = {
            "output_format": "markdown",
            "paginate_output": True,
            "force_ocr": self.force_ocr,
            "disable_image_extraction": True,
        }
        converter = PdfConverter(artifact_dict=self._artifacts(), config=config)
        rendered = converter(str(pdf_path))
        text, _, _ = text_from_rendered(rendered)
        return split_marker_pages(text)


def render_enhanced_pdf(src: Path, dst: Path, dpi: int) -> None:
    """Rasterise each page and apply clean-up filters that help OCR on degraded scans."""
    import pypdfium2 as pdfium
    from PIL import ImageFilter, ImageOps

    pdf = pdfium.PdfDocument(str(src))
    try:
        images = []
        for index in range(len(pdf)):
            image = pdf[index].render(scale=dpi / 72).to_pil().convert("L")
            image = ImageOps.autocontrast(image, cutoff=1)
            image = image.filter(ImageFilter.MedianFilter(3))  # remove speckle noise
            image = image.filter(ImageFilter.SHARPEN)
            images.append(image.convert("RGB"))
    finally:
        pdf.close()
    if not images:
        raise DocumentUnreadableError("The PDF contains no pages.")
    images[0].save(dst, "PDF", save_all=True, append_images=images[1:], resolution=float(dpi))


class EnhancedOcrMarkerStrategy(MarkerStrategy):
    name = "marker_enhanced_ocr"

    def __init__(self, dpi: int = 200) -> None:
        super().__init__(force_ocr=True)
        self.dpi = dpi

    def is_available(self) -> bool:
        return super().is_available() and _has_module("pypdfium2")

    def extract(self, pdf_path: Path) -> list[ParsedPage]:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            enhanced = Path(tmp) / "enhanced.pdf"
            render_enhanced_pdf(pdf_path, enhanced, self.dpi)
            return super().extract(enhanced)


class UnstructuredStrategy:
    """Unstructured.io partitioning (requires tesseract + poppler for ``hi_res``)."""

    name = "unstructured"

    def __init__(self, strategy: str = "hi_res") -> None:
        self.strategy = strategy

    def is_available(self) -> bool:
        return _has_module("unstructured")

    def extract(self, pdf_path: Path) -> list[ParsedPage]:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=str(pdf_path), strategy=self.strategy, infer_table_structure=True
        )
        by_page: dict[int, list[str]] = {}
        for element in elements:
            category = getattr(element, "category", "")
            text = (getattr(element, "text", "") or "").strip()
            if not text or category in {"Header", "Footer", "PageBreak", "Image"}:
                continue
            page = getattr(element.metadata, "page_number", None) or 1
            if category == "Title":
                md = f"## {text}"
            elif category == "ListItem":
                md = f"- {text}"
            elif category == "Table":
                md = getattr(element.metadata, "text_as_html", None) or text
            else:
                md = text
            by_page.setdefault(page, []).append(md)
        return [ParsedPage(n, "\n\n".join(parts)) for n, parts in sorted(by_page.items())]


_KEYWORD_HEADING_RE = re.compile(
    r"^(?:ARTICLE|Article|SECTION|Section|SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)"
    r"\s+[0-9IVXLC]+[A-Z]?\b"
)
_NUMBERED_HEADING_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?\s+([A-Z][^\n]{1,80})$")
_ENDS_SENTENCE_RE = re.compile(r"[.:;]\s*$")
_STARTS_BLOCK_RE = re.compile(r"^(?:\(?[a-z0-9]{1,3}[.)]\s|\d{1,3}(?:\.\d{1,3})+\s|[A-Z•\-])")


def _heading_level(line: str) -> int | None:
    if len(line) > 100 or line.endswith((",", ";")):
        return None
    if line.endswith(".") and len(line.split()) > 4:  # a numbered sentence, not a heading
        return None
    if _KEYWORD_HEADING_RE.match(line):
        return 2
    match = _NUMBERED_HEADING_RE.match(line)
    if match and len(match.group(2).split()) <= 8:
        return min(6, 2 + match.group(1).count("."))
    letters = [c for c in line if c.isalpha()]
    if len(letters) >= 4 and len(line) <= 80 and all(c.isupper() for c in letters):
        return 2
    return None


def text_layer_to_markdown(text: str) -> str:
    """Rebuild paragraphs and headings from a raw PDF text layer (hard-wrapped lines)."""
    blocks: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph))
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        level = _heading_level(line)
        if level is not None:
            flush()
            blocks.append(f"{'#' * level} {line}")
            continue
        if paragraph and _ENDS_SENTENCE_RE.search(paragraph[-1]) and _STARTS_BLOCK_RE.match(line):
            flush()
        paragraph.append(line)
    flush()
    return "\n\n".join(blocks)


class PyPdfTextStrategy:
    """Last resort: the embedded text layer only (no OCR). Fast and dependency-free."""

    name = "pypdf"

    def is_available(self) -> bool:
        return True

    def extract(self, pdf_path: Path) -> list[ParsedPage]:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path), strict=False)
        return [
            ParsedPage(i + 1, text_layer_to_markdown(page.extract_text() or ""))
            for i, page in enumerate(reader.pages)
        ]


def build_strategies(settings: Settings) -> list[ExtractionStrategy]:
    factories = {
        "marker": lambda: MarkerStrategy(),
        "marker_enhanced_ocr": lambda: EnhancedOcrMarkerStrategy(dpi=settings.ocr_render_dpi),
        "unstructured": lambda: UnstructuredStrategy(),
        "pypdf": lambda: PyPdfTextStrategy(),
    }
    unknown = [name for name in settings.parser_strategies if name not in factories]
    if unknown:
        raise ValueError(f"Unknown parser strategies: {unknown}. Valid: {sorted(factories)}")
    return [factories[name]() for name in settings.parser_strategies]


# ---------------------------------------------------------------------------
# DocumentParser
# ---------------------------------------------------------------------------
_IMAGE_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"II*\x00": "tiff",
    b"MM\x00*": "tiff",
}


_OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_WORD_STREAM_NAME = "WordDocument".encode("utf-16-le")


def _is_docx(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False


def detect_file_type(path: Path) -> str:
    """Detect by content, never by extension."""
    with path.open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF-" in head:
        return "pdf"
    for signature in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return "image"
    # Legacy .doc is an OLE2 container (shared with .xls/.ppt); require its WordDocument stream.
    if head.startswith(_OLE2_SIGNATURE) and _WORD_STREAM_NAME in path.read_bytes():
        return "word"
    if head.startswith(b"PK\x03\x04") and _is_docx(path):
        return "word"
    raise UnsupportedFileTypeError(
        "Supported formats: PDF, Word (.doc, .docx) and scanned images (PNG, JPEG, TIFF)."
    )


class DocumentParser:
    def __init__(
        self,
        strategies: Sequence[ExtractionStrategy],
        *,
        quality_threshold: float = 0.6,
        min_quality: float = 0.15,
        min_chars_per_page: int = 200,
        max_pages: int = 500,
        office_converter: OfficeConverter | None = None,
    ) -> None:
        self._strategies = list(strategies)
        self._office = office_converter or OfficeConverter()
        self.quality_threshold = quality_threshold
        self.min_quality = min_quality
        self.min_chars_per_page = min_chars_per_page
        self.max_pages = max_pages

    @classmethod
    def from_settings(cls, settings: Settings) -> "DocumentParser":
        return cls(
            build_strategies(settings),
            quality_threshold=settings.parse_quality_threshold,
            min_quality=settings.parse_min_quality,
            min_chars_per_page=settings.min_chars_per_page,
            max_pages=settings.max_pages,
        )

    def available_strategies(self) -> list[str]:
        return [s.name for s in self._strategies if s.is_available()]

    def available_office_backends(self) -> list[str]:
        return self._office.available_backends()

    # -- Step 1: validation, decryption, image conversion (fast; runs before the 202 response) --
    def prepare(self, src: Path, work_dir: Path, password: str | None = None) -> PreparedDocument:
        file_type = detect_file_type(src)
        if file_type == "image":
            return self._prepare_image(src, work_dir)
        if file_type == "word":
            return self._prepare_word(src, work_dir, password)
        return self._prepare_pdf(src, work_dir, password)

    def _prepare_word(self, src: Path, work_dir: Path, password: str | None) -> PreparedDocument:
        out = work_dir / "converted.pdf"
        self._office.to_pdf(src, out, password)
        prepared = self._prepare_pdf(out, work_dir, None)
        return PreparedDocument(
            out, is_temporary=True, source_type="word", page_count=prepared.page_count, warnings=prepared.warnings
        )

    def _prepare_image(self, src: Path, work_dir: Path) -> PreparedDocument:
        from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

        out = work_dir / "converted.pdf"
        try:
            with Image.open(src) as image:
                frames = [ImageOps.exif_transpose(f).convert("RGB") for f in ImageSequence.Iterator(image)]
        except (UnidentifiedImageError, OSError) as exc:
            raise DocumentUnreadableError("The image file is corrupted or cannot be decoded.") from exc
        if len(frames) > self.max_pages:
            raise TooManyPagesError(f"The image has {len(frames)} pages; the limit is {self.max_pages}.")
        frames[0].save(out, "PDF", save_all=True, append_images=frames[1:], resolution=300.0)
        return PreparedDocument(out, is_temporary=True, source_type="image", page_count=len(frames))

    def _prepare_pdf(self, src: Path, work_dir: Path, password: str | None) -> PreparedDocument:
        from pypdf import PdfReader, PdfWriter
        from pypdf.errors import DependencyError

        try:
            reader = PdfReader(str(src), strict=False)
            encrypted = reader.is_encrypted
        except Exception as exc:  # damaged xref/trailer: let the OCR strategies (pdfium) try anyway
            logger.warning("pypdf could not open %s: %s", src.name, exc)
            return PreparedDocument(
                src,
                is_temporary=False,
                source_type="pdf",
                page_count=None,
                warnings=["The PDF structure is damaged; extraction used recovery mode."],
            )

        if encrypted:
            try:
                unlocked = reader.decrypt(password or "")
            except DependencyError as exc:
                raise LegalBotError(
                    "This PDF uses AES encryption; install 'cryptography' on the server to open it."
                ) from exc
            except Exception as exc:
                raise DocumentUnreadableError("The PDF's encryption could not be processed.") from exc
            if not unlocked:  # PasswordType.NOT_DECRYPTED == 0
                if password:
                    raise InvalidPasswordError("The password supplied for this PDF is incorrect.")
                raise PasswordRequiredError("This PDF is password-protected. Provide its password to continue.")

        try:
            page_count = len(reader.pages)
        except Exception:
            page_count = None
        if page_count is not None and page_count > self.max_pages:
            raise TooManyPagesError(f"The PDF has {page_count} pages; the limit is {self.max_pages}.")
        if page_count == 0:
            raise DocumentUnreadableError("The PDF contains no pages.")

        if not encrypted:
            return PreparedDocument(src, is_temporary=False, source_type="pdf", page_count=page_count)

        # Write a decrypted working copy for the extraction libraries; deleted after parsing.
        out = work_dir / "decrypted.pdf"
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        with out.open("wb") as handle:
            writer.write(handle)
        return PreparedDocument(out, is_temporary=True, source_type="pdf", page_count=page_count)

    # -- Step 2: extraction with fallbacks (slow; runs in the background) --
    def parse(self, prepared: PreparedDocument) -> ParsedDocument:
        warnings = list(prepared.warnings)
        failures: list[str] = []
        best: ParsedDocument | None = None

        for strategy in self._strategies:
            if not strategy.is_available():
                continue
            try:
                pages = strategy.extract(prepared.path)
            except Exception as exc:
                logger.warning("Extraction strategy %s failed: %s", strategy.name, exc, exc_info=True)
                failures.append(f"{strategy.name}: {type(exc).__name__}")
                continue

            pages = [ParsedPage(p.number, clean_markdown(p.markdown)) for p in pages]
            quality = assess_quality(pages, self.min_chars_per_page)
            logger.info("Strategy %s scored %.3f", strategy.name, quality.score)
            if best is None or quality.score > best.quality_score:
                best = ParsedDocument(pages, strategy.name, quality.score, quality.low_text_pages)
            if quality.score >= self.quality_threshold:
                break

        if best is None:
            detail = "; ".join(failures) or "no extraction strategy is installed"
            raise DocumentUnreadableError(f"The document could not be read ({detail}).")
        if best.quality_score < self.min_quality:
            raise DocumentUnreadableError(
                "The document is too degraded to extract reliable text. "
                "Try a higher-resolution scan or the original digital PDF.",
                details={"quality_score": round(best.quality_score, 3), "parser": best.parser_used},
            )

        if failures:
            warnings.append(f"Some extraction methods failed: {'; '.join(failures)}.")
        if best.quality_score < self.quality_threshold:
            warnings.append(
                "Extraction quality is low (degraded or unusual scan). "
                "Answers may miss or misread parts of the document."
            )
        if best.low_text_pages:
            pages = ", ".join(map(str, best.low_text_pages[:20]))
            more = "..." if len(best.low_text_pages) > 20 else ""
            warnings.append(f"Little or no readable text on page(s) {pages}{more}.")
        best.warnings = warnings
        return best
