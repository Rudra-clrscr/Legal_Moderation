from app.services.chunker import MarkdownChunker, split_numbered_title
from app.services.document_parser import ParsedPage

LONG_41 = (
    "**4.1** Either party may terminate this Agreement for convenience by giving the other party "
    "not less than thirty (30) days prior written notice, provided that all undisputed fees accrued "
    "up to the effective date of termination remain payable in full."
)


def test_split_numbered_title():
    assert split_numbered_title("4.1 Termination") == ("Section 4.1", "Termination")
    assert split_numbered_title("4. TERMINATION") == ("Section 4", "TERMINATION")
    assert split_numbered_title("ARTICLE IV - Liability") == ("Article IV", "Liability")
    assert split_numbered_title("Confidentiality") == (None, "Confidentiality")
    assert split_numbered_title("2024 Pricing Update") == (None, "2024 Pricing Update")


def test_chunks_follow_headings_clauses_and_pages():
    pages = [
        ParsedPage(1, "# MASTER AGREEMENT\n\nThis Agreement is made between Acme Ltd and Beta LLC.\n\n"
                      f"## 4. TERMINATION\n\n{LONG_41}"),
        ParsedPage(2, "The notice must be delivered in writing to the registered address.\n\n"
                      "**4.2** The Supplier may terminate immediately if the Customer fails to pay any invoice "
                      "within sixty days of its due date, without any liability to the Customer.\n\n"
                      "## 5. LIABILITY\n\n### 5.1 Cap\n\nLiability is capped at the fees paid in the prior 12 months."),
    ]
    chunks = MarkdownChunker(target_tokens=120, max_tokens=200, min_tokens=30).chunk("doc", pages)
    by_label = {c.section_label: c for c in chunks}

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert by_label["MASTER AGREEMENT"].header_path == ["MASTER AGREEMENT"]

    s41 = by_label["Section 4.1"]
    assert s41.section_title == "TERMINATION"
    assert (s41.page_start, s41.page_end) == (1, 2)  # continuation paragraph inherits clause 4.1
    assert "registered address" in s41.text

    s42 = by_label["Section 4.2"]
    assert s42.page_start == 2 and "fails to pay" in s42.text

    s51 = by_label["Section 5.1"]
    assert s51.header_path == ["MASTER AGREEMENT", "5. LIABILITY", "5.1 Cap"]
    assert s51.section_title == "Cap"
    assert s51.embedding_text.startswith("MASTER AGREEMENT > 5. LIABILITY > 5.1 Cap")


def test_short_clauses_merge_into_range_label():
    pages = [ParsedPage(1, "## 3. TERM\n\n3.1 Term is one year.\n\n3.2 Renewal is automatic.")]
    chunks = MarkdownChunker(target_tokens=300, max_tokens=400, min_tokens=60).chunk("doc", pages)
    assert len(chunks) == 1
    assert chunks[0].section_label == "Sections 3.1–3.2"


def test_oversized_paragraph_is_split_within_limits():
    sentence = "The Licensee shall indemnify the Licensor against all claims arising from use of the Software. "
    pages = [ParsedPage(3, "## 9. INDEMNITY\n\n" + sentence * 60)]
    chunker = MarkdownChunker(target_tokens=100, max_tokens=150, min_tokens=20)
    chunks = chunker.chunk("doc", pages)
    assert len(chunks) > 5
    assert all(chunker.count(c.text) <= 150 for c in chunks)
    assert all(c.section_label == "Section 9" and c.page_start == 3 for c in chunks)


def test_metadata_round_trip():
    chunk = MarkdownChunker().chunk("abc", [ParsedPage(2, "## 7.2 Fees\n\nFees are payable monthly.")])[0]
    from app.services.chunker import Chunk

    restored = Chunk.from_record(chunk.chunk_id, chunk.text, chunk.to_metadata())
    assert restored == chunk
    assert all(isinstance(v, (str, int, float, bool)) for v in chunk.to_metadata().values())


def test_number_only_heading_merges_with_next_heading():
    pages = [ParsedPage(11, "## 12.\n\n## INDEMNITY\n\n1. The Partner shall indemnify the Bank.")]
    chunk = MarkdownChunker().chunk("doc", pages)[0]
    assert chunk.header_path == ["12. INDEMNITY"]
    assert chunk.section_label == "Section 12" and chunk.section_title == "INDEMNITY"


def test_flat_clause_numbers_only_without_numbered_heading():
    body = "Each party shall keep the other party's information confidential at all times. " * 3
    flat = [ParsedPage(1, f"## PREAMBLE:\n\n1. {body}\n\n2. {body}")]
    labels = [c.section_label for c in MarkdownChunker(min_tokens=20).chunk("d", flat)]
    assert labels == ["Section 1", "Section 2"]

    nested = [ParsedPage(1, f"## 12. INDEMNITY\n\n1. {body}\n\n2. {body}")]
    labels = {c.section_label for c in MarkdownChunker(min_tokens=20).chunk("d", nested)}
    assert labels == {"Section 12"}


def test_list_formatted_clauses_without_blank_lines_are_split():
    body = "The Receiving Party shall protect all Confidential Information with reasonable care. " * 2
    pages = [ParsedPage(3, f"## PREAMBLE:\n\n4. {body}\n5. {body}\n- 6. {body}")]
    labels = [c.section_label for c in MarkdownChunker(min_tokens=20).chunk("d", pages)]
    assert labels == ["Section 4", "Section 5", "Section 6"]
