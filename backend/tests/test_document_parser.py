from pathlib import Path

import pytest

from app.errors import (
    DocumentUnreadableError,
    InvalidPasswordError,
    PasswordRequiredError,
    UnsupportedFileTypeError,
)
from app.services.document_parser import (
    DocumentParser,
    ParsedPage,
    PyPdfTextStrategy,
    assess_quality,
    split_marker_pages,
    text_layer_to_markdown,
)
from tests.conftest import encrypt_pdf


class FakeStrategy:
    def __init__(self, name, pages=None, error=None, available=True):
        self.name, self._pages, self._error, self._available = name, pages, error, available
        self.calls = 0

    def is_available(self):
        return self._available

    def extract(self, pdf_path):
        self.calls += 1
        if self._error:
            raise self._error
        return self._pages


GOOD_TEXT = "The Customer shall pay all invoices within thirty days of receipt. " * 6
GARBAGE = "l1|; ~~ ^^ |l| ;;:: 0O0 ]]][ " * 8


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_quality_scores_separate_text_from_garbage():
    good = assess_quality([ParsedPage(1, GOOD_TEXT)], 200).score
    bad = assess_quality([ParsedPage(1, GARBAGE)], 200).score
    empty = assess_quality([ParsedPage(1, "")], 200)
    assert good > 0.9 and bad < 0.2
    assert empty.score == 0 and empty.low_text_pages == [1]


def test_fallback_chain_skips_failures_and_stops_at_good_result(tmp_path, contract_pdf):
    first = FakeStrategy("marker", error=RuntimeError("model crash"))
    second = FakeStrategy("marker_enhanced_ocr", pages=[ParsedPage(1, GARBAGE)])
    third = FakeStrategy("unstructured", pages=[ParsedPage(1, GOOD_TEXT)])
    fourth = FakeStrategy("pypdf", pages=[ParsedPage(1, GOOD_TEXT)])
    unavailable = FakeStrategy("missing", available=False)
    parser = DocumentParser([unavailable, first, second, third, fourth])

    prepared = parser.prepare(write(tmp_path, "a.pdf", contract_pdf), tmp_path)
    result = parser.parse(prepared)

    assert result.parser_used == "unstructured"
    assert fourth.calls == 0 and unavailable.calls == 0
    assert any("marker: RuntimeError" in w for w in result.warnings)


def test_degraded_document_is_rejected(tmp_path, contract_pdf):
    parser = DocumentParser([FakeStrategy("ocr", pages=[ParsedPage(1, GARBAGE)])])
    prepared = parser.prepare(write(tmp_path, "a.pdf", contract_pdf), tmp_path)
    with pytest.raises(DocumentUnreadableError):
        parser.parse(prepared)


def test_low_quality_result_is_accepted_with_warning(tmp_path, contract_pdf):
    mixed = [ParsedPage(1, GOOD_TEXT), ParsedPage(2, ""), ParsedPage(3, GOOD_TEXT[:150])]
    parser = DocumentParser([FakeStrategy("ocr", pages=mixed)], quality_threshold=0.95)
    result = parser.parse(parser.prepare(write(tmp_path, "a.pdf", contract_pdf), tmp_path))
    assert result.low_text_pages == [2]
    assert any("quality is low" in w for w in result.warnings)
    assert any("page(s) 2" in w for w in result.warnings)


def test_password_protected_pdf(tmp_path, contract_pdf):
    src = write(tmp_path, "locked.pdf", encrypt_pdf(contract_pdf, "s3cret"))
    parser = DocumentParser([PyPdfTextStrategy()])

    with pytest.raises(PasswordRequiredError):
        parser.prepare(src, tmp_path)
    with pytest.raises(InvalidPasswordError):
        parser.prepare(src, tmp_path, password="wrong")

    prepared = parser.prepare(src, tmp_path, password="s3cret")
    assert prepared.is_temporary and prepared.path != src
    result = parser.parse(prepared)
    assert "Supplier may terminate" in result.markdown


def test_image_is_converted_to_pdf(tmp_path):
    from PIL import Image

    img_path = tmp_path / "scan.png"
    Image.new("RGB", (200, 300), "white").save(img_path)
    prepared = DocumentParser([PyPdfTextStrategy()]).prepare(img_path, tmp_path)
    assert prepared.source_type == "image" and prepared.page_count == 1
    assert prepared.path.read_bytes().startswith(b"%PDF")


def test_unsupported_file_type(tmp_path):
    with pytest.raises(UnsupportedFileTypeError):
        DocumentParser([]).prepare(write(tmp_path, "x.pdf", b"PK\x03\x04 not a pdf"), tmp_path)


def test_pypdf_text_layer_headings(tmp_path, contract_pdf):
    pages = PyPdfTextStrategy().extract(write(tmp_path, "c.pdf", contract_pdf))
    assert len(pages) == 2
    assert "## 3. TERMINATION" in pages[1].markdown
    assert "## MASTER SERVICES AGREEMENT" in pages[0].markdown
    # Numbered sentences stay paragraphs, not headings:
    assert "\n\n3.1 The Supplier may terminate" in pages[1].markdown


def test_text_layer_rejoins_wrapped_lines():
    md = text_layer_to_markdown("1. TERM\nThe term of this Agreement is\ntwelve months from signature.")
    assert md == "## 1. TERM\n\nThe term of this Agreement is twelve months from signature."


def test_split_marker_pages():
    text = "{0}" + "-" * 48 + "\n\n# Title\n\nIntro\n\n{1}" + "-" * 48 + "\n\nSecond page"
    pages = split_marker_pages(text)
    assert [(p.number, p.markdown.strip()) for p in pages] == [(1, "# Title\n\nIntro"), (2, "Second page")]


class FakeOffice:
    def __init__(self, pdf_bytes):
        self.pdf_bytes, self.calls = pdf_bytes, []

    def available_backends(self):
        return ["fake"]

    def to_pdf(self, src, dst, password=None):
        self.calls.append(password)
        dst.write_bytes(self.pdf_bytes)


def test_word_documents_are_converted(tmp_path, contract_pdf):
    import zipfile

    docx = tmp_path / "contract.docx"
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", "<w:document/>")
    ole_doc = write(tmp_path, "legacy.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64 + "WordDocument".encode("utf-16-le"))
    office = FakeOffice(contract_pdf)
    parser = DocumentParser([PyPdfTextStrategy()], office_converter=office)

    for src in (docx, ole_doc):
        prepared = parser.prepare(src, tmp_path)
        assert prepared.source_type == "word" and prepared.is_temporary and prepared.page_count == 2
    assert "Supplier may terminate" in parser.parse(prepared).markdown


def test_non_word_ole2_and_zip_rejected(tmp_path):
    import zipfile

    xls = write(tmp_path, "sheet.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + "Workbook".encode("utf-16-le"))
    other_zip = tmp_path / "x.zip"
    with zipfile.ZipFile(other_zip, "w") as archive:
        archive.writestr("readme.txt", "hi")
    for path in (xls, other_zip):
        with pytest.raises(UnsupportedFileTypeError):
            DocumentParser([]).prepare(path, tmp_path)


def test_clean_markdown_collapses_padded_tables_and_trivial_math():
    from app.services.document_parser import clean_markdown

    md = "## 12. $-$\n\n| 31" + " " * 200 + "| text here |" + " " * 50 + "|  |  |\n|" + "-" * 120 + "|----|\n"
    cleaned = clean_markdown(md)
    assert "$" not in cleaned
    assert "  " not in cleaned
    assert len(cleaned) < 60


def test_clean_markdown_turns_ocr_text_math_back_into_text():
    from app.services.document_parser import clean_markdown

    # Marker OCR output for small or spaced-out lines on scanned resumes.
    md = (
        "QA Engineer Zilkalns $\\mathrm { ~ \\scriptsize ~ O c t ~ 2 0 2 2 - M a y ~ 2 0 2 4 }$ Riga\n"
        "mail@example.com · +1 $5 5 5 - o 1 6 7 \\cdot \\mathrm { R i g a , L a t v i a \\cdot D O B : 2 o }$ Apr 1996"
    )
    assert clean_markdown(md) == (
        "QA Engineer Zilkalns Oct 2022-May 2024 Riga\n"
        "mail@example.com · +1 555-0167 · Riga, Latvia · DOB: 20 Apr 1996"
    )
    # Real formulas are not touched.
    for formula in ("$E = mc^2$", "$x_1 + x_2$", "$\\frac { a } { b }$"):
        assert formula in clean_markdown(f"see {formula} here")
