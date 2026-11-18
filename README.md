# Legal Document Analysis Chatbot

A local RAG backend that reads contracts (including scanned ones), answers questions, flags risks with
validated citations, and produces map-reduce summaries. Everything runs on your machine: Marker OCR,
`bge-small-en-v1.5` embeddings, ChromaDB, and a 4-bit ONNX 7B model through `onnxruntime-genai`.

## Layout

```
demo.html                    interactive demo page (standalone root preview)
backend/
  app/
    main.py                  FastAPI app factory + startup warm-up
    container.py             builds and wires the services
    config.py                settings (env vars, prefix LEGAL_)
    api/routes.py            REST + SSE endpoints
    services/
      document_parser.py     DocumentParser: validation, decryption, OCR fallback chain -> Markdown pages
      chunker.py             MarkdownChunker: heading/clause-aware chunks with page + section metadata
      vector_store.py        Embedder (bge-small) + VectorStore (ChromaDB)
      llm_generator.py       LLMGenerator: onnxruntime-genai with async token streaming
      prompts.py             system/task prompts (analytical framing, no legal advice)
      citations.py           [S#] citation validation, risk-finding parser
      analysis_service.py    Q&A and risk-analysis RAG pipelines
      summarizer.py          map-reduce summaries + background jobs
      ingestion_service.py   upload -> parse -> chunk -> embed pipeline
      document_registry.py   file-backed document metadata
    static/
      index.html             interactive demo & reference UI
      demo.html              dedicated demo UI endpoint
  tests/
```

The parser has no dependency on the LLM or vector store, and the LLM layer knows nothing about documents.
The services meet only in `container.py`.

## Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ocr.txt    # Marker/Surya OCR (PyTorch, large download)
copy ..\.env.example .env               # optional; edit as needed
```

On Windows, keep the virtual environment on a short path. PyTorch contains files that hit the
260-character `MAX_PATH` limit inside deep directories.

**Word files** (`.doc`, `.docx`) are converted to PDF with LibreOffice (`soffice` on `PATH`) or, on
Windows, through Microsoft Word (`pywin32`, included in `requirements.txt`). Page numbers in
citations therefore match the document's own pagination. `GET /api/health` lists which converter is
available.

### Get an ONNX model

Convert with the `onnxruntime-genai` model builder (`-e cpu`, `-e cuda` or `-e dml`). Use a
separate virtual environment for conversion: it needs `transformers` and `torch`, whose versions can
clash with Marker's.

```powershell
python -m venv convert-venv
convert-venv\Scripts\pip install onnxruntime-genai onnx onnx_ir transformers torch sentencepiece protobuf accelerate
```

**Recommended: start from Qwen's official AWQ checkpoint** (`Qwen/Qwen2.5-7B-Instruct-AWQ`, 5.6 GB).
The builder re-packs the AWQ int4 weights directly. That is a third of the full-precision download,
needs little RAM, and AWQ usually keeps more quality than the builder's default int4 quantization.

```powershell
hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir qwen2.5-7b-awq
convert-venv\Scripts\python -m onnxruntime_genai.models.builder -i qwen2.5-7b-awq -o backend/models/qwen2.5-7b-instruct-awq-onnx-int4 -p int4 -e cpu -c hf_cache
```

Alternatives that quantize from full-precision weights (about 15 GB download, about 2× the model size
in RAM while converting):

```powershell
# Qwen2.5-7B-Instruct (ChatML template)
python -m onnxruntime_genai.models.builder -m Qwen/Qwen2.5-7B-Instruct -o backend/models/qwen2.5-7b-instruct-onnx-int4 -p int4 -e cpu

# SaulLM-7B-Instruct (set LEGAL_LLM_PROMPT_TEMPLATE=mistral)
python -m onnxruntime_genai.models.builder -m Equall/Saul-7B-Instruct-v1 -o backend/models/saullm-7b-instruct-onnx-int4 -p int4 -e cpu
```

Then point `LEGAL_LLM_MODEL_PATH` at the output folder. For GPU, install `onnxruntime-genai-cuda` or
`onnxruntime-genai-directml` in place of `onnxruntime-genai`. A 7B int4 model with its KV cache
needs more than 6 GB of VRAM for long contract prompts, so on smaller GPUs keep the LLM on the CPU
and let Marker use the GPU.

### Run on Localhost

#### Option 1: Full AI Backend (FastAPI + OCR + Embeddings + ONNX 7B LLM)

Runs the complete local RAG backend with streaming SSE endpoints, ChromaDB vector store, Marker OCR, and the interactive demo UI:

```powershell
cd backend
.\.venv\Scripts\activate
python run_server.py --port 8080
```

*For Linux/macOS:*
```bash
cd backend
source .venv/bin/activate
python run_server.py --port 8080
```

Open **http://localhost:8080** or **http://localhost:8080/demo.html** in your browser.

- `run_server.py` starts uvicorn on port **`8080`** by default in **Python UTF-8 mode** (you can specify any custom port using `--port <PORT>`). Marker's OCR engine (`indra_ocr`) reads its UTF-8 model metadata with the locale codec, which fails on Windows (cp1252) without that mode. If you start uvicorn yourself, use `python -X utf8 -m uvicorn app.main:app --port 8080`. Without it, Marker is skipped with a clear warning and the parser falls back to the text layer, so scanned documents fail.
- `GET /api/health` shows the model state and which parsers are installed. The server starts even without an ONNX model; analysis endpoints then return `503 MODEL_UNAVAILABLE`.

#### Option 2: Instant Standalone Demo (No Model Load Time)

To preview and test the interactive demo page without loading or waiting for the local 7B model (includes preloaded sample contracts, simulated streaming Q&A, risk matrices, and citation popovers):

```powershell
# From the repository root
python -m http.server 8080
```

Open **http://localhost:8080/demo.html** in your browser.

### Tests

```powershell
pip install -r requirements-dev.txt
pytest
```

The tests use stub LLM and vector-store implementations, so they don't need the models.

## How it works

### Ingestion (Phase A)
1. **Upload**: `POST /api/documents` with a multipart `file` and an optional `password`. The file type
   is detected from its content (PDF, Word `.doc`/`.docx`, PNG, JPEG, TIFF); Word files and images
   are converted to PDF.
2. **Password check**: done before the response, so the client gets `PASSWORD_REQUIRED` or
   `INVALID_PASSWORD` (422) right away. The decrypted working copy is deleted after parsing.
3. **Extraction chain**: each strategy's output gets a quality score (share of plausible words × text
   density). The first strategy to reach `LEGAL_PARSE_QUALITY_THRESHOLD` wins; otherwise the best
   attempt is kept and flagged:

   | Order | Strategy | Purpose |
   |---|---|---|
   | 1 | `marker` | Layout-aware Markdown with OCR for scanned pages |
   | 2 | `marker_enhanced_ocr` | Re-renders pages, applies contrast/denoise/sharpen, forces OCR (degraded scans) |
   | 3 | `unstructured` | Unstructured.io `hi_res` (optional install) |
   | 4 | `pypdf` | Text layer only, with heuristic headings (always available) |

   Below `LEGAL_PARSE_MIN_QUALITY` the document fails with `DOCUMENT_UNREADABLE`. Warnings such as
   low-quality extraction or pages with no readable text are stored in `parse_report` and shown in the UI.
4. **Chunking**: chunks never cross a heading, and a new chunk starts at each numbered clause (`4.1`,
   `4.2`). Only oversized paragraphs are split further (lines, then sentences, then words). Each chunk
   carries `section_label` ("Section 4.1"), `section_title`, `header_path` and its page range.
5. **Embedding**: the header path plus the text is embedded with bge-small and stored in ChromaDB.

### Q&A and risk analysis (Phase B)
- `POST /api/documents/{id}/chat` with `{"question": "..."}`
- `POST /api/documents/{id}/risk-analysis` with `{"party": "Customer", "focus": "termination"}`
  (both fields optional). Risk analysis retrieves with several clause-family queries (liability,
  indemnity, termination, auto-renewal and others), interleaved so each topic gets represented.

Both endpoints return Server-Sent Events:

| Event | Payload |
|---|---|
| `sources` | every excerpt sent to the LLM: `source_id`, exact `text`, `section_label`, pages, `citation` ("Source: Section 4.1 — Termination, p. 3") |
| `token` | `{"text": "..."}` streamed output |
| `citations` | the excerpts the answer actually cited (validated), plus `invalid_source_ids` for hallucinated ones |
| `risks` | risk analysis only: `title`, `severity`, `analysis`, `source_ids`, `citation_missing` |
| `done` | `disclaimer`, `uncited` |
| `error` | `code`, `message` |

Excerpts fit within the context window (`context_window − max_new_tokens − prompt`) and are numbered
in document order. The system prompt requires `[S#]` citations, frames output as "legal analysis
based on the provided text", forbids advice-style language, and tells the model to treat document
text as untrusted (a prompt-injection guard). A risk finding without a valid citation is flagged
`citation_missing` instead of being shown as sourced.

### Summaries (Phase C)
`POST /api/documents/{id}/summary` starts a background job (`{"force": true}` regenerates it), and
`GET` on the same path polls its progress. Chunks are packed into about 2,500-token batches, preferring
top-level section boundaries. **Map** summarises each batch; **reduce** writes the executive summary
under fixed headings. If the combined batch summaries don't fit, they are condensed in groups first
(up to 3 levels deep). Results are saved to `data/documents/<id>/summary.json`.

### Other endpoints
`GET /api/documents`, `GET /api/documents/{id}`, `DELETE /api/documents/{id}`,
`GET /api/documents/{id}/chunks` (shows how a document was chunked).

## Measured on the reference laptop

Ryzen 7 7445HS (6 cores), 32 GB RAM, RTX 4050 6 GB. Qwen2.5-7B-Instruct AWQ → ONNX int4, CPU:

| Step | Time |
|---|---|
| Marker OCR / parse (Torch on CPU) | 1–30 s per document; an 18-page scan takes about 12 s |
| Embed and index | 2–5 s per document |
| Model load | about 20 s (about 4.3 GB RAM in the worker process) |
| Q&A answer (6 excerpts) | 80–165 s; 60–120 s of that is prompt processing before the first token |
| Risk analysis (12–13 excerpts) | 3.5–8 min |
| Summary (map-reduce) | 6 min (8-page NDA) to 26 min (23-page MSA) |

The CPU processes prompts at 20–35 tokens/s and generates at about 7 tokens/s. The DirectML build
(`-e dml`) on the 6 GB RTX 4050 processes prompts at about 275 tokens/s, but the model plus KV cache
overflows VRAM, so generation drops to about 3 tokens/s. A GPU with 8 GB or more of VRAM would cut
all LLM times by roughly 10×.

## Notes and limitations
- **Risk analysis is retrieval-based**: it sees about 13 excerpts chosen by 10 risk-category probes,
  so a clause worded unusually can be missed. On the test set, a one-year exclusivity covenant
  ("shall not … implement … with other commercial banks") never reached the prompt. The summary does
  read every chunk and caught it. A map-reduce risk scan over all chunks would close this gap, at
  summary-like cost.
- **The 7B model can omit or misjudge**: for example, it didn't list the MSA's six-month liability cap
  as a disadvantage for the Client even though the clause was in its context. Treat findings as a
  review aid.
- **Data at rest**: `data/` holds the original upload, the extracted Markdown (`parsed.md`) and the
  vectors in plaintext, including for password-protected PDFs. Put it on encrypted storage if needed.
- **No authentication**: any client that can reach the server can see every document. Add auth before
  exposing it beyond localhost.
- **Restarts**: summary jobs live in memory. A restart during processing marks the document `INTERRUPTED`.
- **LLM worker process**: the model runs in a separate process (`llm_worker.py`) because
  `onnxruntime_genai` holds the Python GIL during native calls. In-process, a 60–120 s prompt
  evaluation would freeze the whole API. Generations run one at a time; a long summary shares the
  model with chat, so they interleave call by call, and a chat request may wait for the current
  map step to finish.
- Marker's page splitting relies on its `paginate_output` separator format. If a future Marker release
  changes it, pages collapse to "page 1" (`split_marker_pages` in `document_parser.py`).
