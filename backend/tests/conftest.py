"""Shared fixtures: stub LLM / vector store and a minimal PDF builder (no heavy models needed)."""

from __future__ import annotations

import io
import re
from typing import Sequence

import pytest

from app.services.chunker import Chunk, estimate_tokens
from app.services.llm_generator import ChatMessage, GenerationOptions
from app.services.vector_store import RetrievedChunk


def make_text_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a small valid PDF with a real text layer (Helvetica, one line per entry)."""
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for pid, lines in zip(page_ids, pages):
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode()
        objs[pid + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objs[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objs) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for oid in range(1, size):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def encrypt_pdf(data: bytes, password: str) -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(data)).pages:
        writer.add_page(page)
    writer.encrypt(user_password=password, owner_password=password + "-owner")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


CONTRACT_PAGES = [
    [
        "MASTER SERVICES AGREEMENT",
        "1. DEFINITIONS",
        "1.1 Services means the consulting services described in Schedule A to this Agreement.",
        "2. FEES AND PAYMENT",
        "2.1 The Customer shall pay all invoices within 15 days. Late payments accrue interest at 5% per month.",
    ],
    [
        "3. TERMINATION",
        "3.1 The Supplier may terminate this Agreement at any time for convenience on 7 days notice.",
        "3.2 The Customer may terminate only for material breach after a 90 day cure period.",
        "4. LIABILITY",
        "4.1 The liability of the Customer is unlimited. The liability of the Supplier is capped at fees paid in the prior month.",
    ],
]


@pytest.fixture
def contract_pdf() -> bytes:
    return make_text_pdf(CONTRACT_PAGES)


class StubLLM:
    """Implements the LLMGenerator interface used by the services."""

    def __init__(self, reply: str = "Based on the provided text, the Supplier may terminate on 7 days notice [S1].") -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []
        self.context_window = 8192

    status = {"state": "ready", "error": None, "model_path": "stub"}

    async def ensure_ready(self) -> None:
        return None

    def render_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return "\n".join(m.content for m in messages)

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    async def stream(self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None):
        self.calls.append(list(messages))
        for piece in re.split(r"(\s+)", self.reply):
            if piece:
                yield piece

    async def generate(self, messages: Sequence[ChatMessage], options: GenerationOptions | None = None) -> str:
        return "".join([p async for p in self.stream(messages, options)])


class StubVectorStore:
    """In-memory store ranking by word overlap."""

    def __init__(self) -> None:
        self.chunks: dict[str, list[Chunk]] = {}

    class _Embedder:
        def warm_up(self) -> None: ...

    embedder = _Embedder()

    async def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self.chunks.setdefault(chunk.doc_id, []).append(chunk)

    async def delete_document(self, doc_id: str) -> None:
        self.chunks.pop(doc_id, None)

    def _rank(self, doc_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in self.chunks.get(doc_id, []):
            overlap = len(words & set(re.findall(r"\w+", chunk.embedding_text.lower())))
            scored.append(RetrievedChunk(chunk, score=overlap / max(1, len(words))))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def search(self, doc_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._rank(doc_id, query, top_k)

    async def search_many(self, doc_id: str, queries: Sequence[str], top_k: int):
        return [self._rank(doc_id, q, top_k) for q in queries]

    async def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        return sorted(self.chunks.get(doc_id, []), key=lambda c: c.index)
