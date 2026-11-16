"""End-to-end API tests: real parser (pypdf strategy), chunker, registry and services;
stub vector store and LLM."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container
from app.main import create_app
from app.services.analysis_service import AnalysisService
from app.services.chunker import MarkdownChunker
from app.services.document_parser import DocumentParser, PyPdfTextStrategy
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService
from app.services.summarizer import DocumentSummarizer, SummaryJobManager
from tests.conftest import StubLLM, StubVectorStore, encrypt_pdf, make_text_pdf

RISK_REPLY = """### Risk 1: Supplier termination for convenience
**Severity:** High
**Source:** [S3]
**Analysis:** The Supplier may terminate "at any time for convenience".

### Overall Observations
Termination rights are asymmetric [S3]."""


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path, llm_load_on_startup=False)
    settings.tmp_dir.mkdir(parents=True)
    registry = DocumentRegistry(settings.documents_dir)
    parser = DocumentParser([PyPdfTextStrategy()])
    vector_store, llm = StubVectorStore(), StubLLM()
    chunker = MarkdownChunker(target_tokens=100, max_tokens=200, min_tokens=5)
    container = Container(
        settings=settings,
        registry=registry,
        parser=parser,
        vector_store=vector_store,
        llm=llm,
        ingestion=IngestionService(registry, parser, chunker, vector_store),
        analysis=AnalysisService(vector_store, llm, settings),
        summaries=SummaryJobManager(DocumentSummarizer(vector_store, llm, settings), registry),
    )
    with TestClient(create_app(container=container)) as client:
        yield client, container


def wait_for(client, url, predicate, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(url).json()
        if predicate(body):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting on {url}: {body}")


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def upload(client, data, password=None):
    form = {"password": password} if password else {}
    return client.post("/api/documents", files={"file": ("contract.pdf", data, "application/pdf")}, data=form)


def test_upload_chat_risks_summary_flow(env, contract_pdf):
    client, container = env
    res = upload(client, contract_pdf)
    assert res.status_code == 202, res.text
    doc_id = res.json()["doc_id"]

    doc = wait_for(client, f"/api/documents/{doc_id}", lambda d: d["status"] != "processing")
    assert doc["status"] == "ready", doc
    assert doc["parse_report"]["parser_used"] == "pypdf"
    assert doc["parse_report"]["page_count"] == 2

    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    labels = [c["section_label"] for c in chunks]
    assert "Section 3.1" in labels and "Section 4.1" in labels

    # --- Q&A stream ---------------------------------------------------------
    res = client.post(f"/api/documents/{doc_id}/chat", json={"question": "Can the Supplier terminate for convenience?"})
    assert res.status_code == 200
    events = parse_sse(res.text)
    names = [e for e, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and "token" in names
    sources = events[0][1]["sources"]
    assert [s["source_id"] for s in sources] == [f"S{i}" for i in range(1, len(sources) + 1)]
    citations = dict(events)["citations"]
    assert [c["source_id"] for c in citations["citations"]] == ["S1"]
    assert citations["citations"][0]["citation"].startswith("Source: Section")
    assert "legal advice" in dict(events)["done"]["disclaimer"]
    system_prompt = container.llm.calls[-1][0].content
    assert "legal analysis based on the provided text" in system_prompt and "Do not give legal advice" in system_prompt

    # --- Risk analysis -------------------------------------------------------------
    container.llm.reply = RISK_REPLY
    res = client.post(f"/api/documents/{doc_id}/risk-analysis", json={"party": "Customer"})
    events = dict(parse_sse(res.text))
    risk = events["risks"]["risks"][0]
    assert risk["severity"] == "High" and risk["source_ids"] == ["S3"] and not risk["citation_missing"]
    assert "for the Customer" in container.llm.calls[-1][1].content or "Customer" in container.llm.calls[-1][1].content

    # --- Summary (map-reduce job) ------------------------------------------------------
    container.llm.reply = "- Summary bullet (Section 1.1)"
    res = client.post(f"/api/documents/{doc_id}/summary", json={})
    assert res.status_code == 202
    job = wait_for(client, f"/api/documents/{doc_id}/summary", lambda j: j["status"] != "running")
    assert job["status"] == "completed", job
    assert job["summary"] and job["section_summaries"]
    assert job["progress_done"] == job["progress_total"]

    # --- Delete -------------------------------------------------------------------------
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert doc_id not in container.vector_store.chunks


def test_password_errors_are_immediate(env, contract_pdf):
    client, _ = env
    locked = encrypt_pdf(contract_pdf, "pw123")
    res = upload(client, locked)
    assert res.status_code == 422 and res.json()["error"]["code"] == "PASSWORD_REQUIRED"
    res = upload(client, locked, password="nope")
    assert res.json()["error"]["code"] == "INVALID_PASSWORD"
    res = upload(client, locked, password="pw123")
    assert res.status_code == 202
    doc = wait_for(client, f"/api/documents/{res.json()['doc_id']}", lambda d: d["status"] != "processing")
    assert doc["status"] == "ready"
    doc_dir = client.app.state.container.registry.doc_dir(doc["doc_id"])
    assert not (doc_dir / "decrypted.pdf").exists()  # decrypted working copy removed


def test_scanned_pdf_without_ocr_fails_cleanly(env):
    client, _ = env
    res = upload(client, make_text_pdf([[], []]))  # pages with no text layer, no OCR strategy configured
    assert res.status_code == 202
    doc = wait_for(client, f"/api/documents/{res.json()['doc_id']}", lambda d: d["status"] != "processing")
    assert doc["status"] == "failed" and doc["error"]["code"] == "DOCUMENT_UNREADABLE"


def test_analysis_requires_ready_document(env):
    client, _ = env
    assert client.post("/api/documents/" + "0" * 32 + "/chat", json={"question": "x"}).status_code == 404
    assert client.get("/api/documents/../../etc").status_code == 404
    res = client.post("/api/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 415
