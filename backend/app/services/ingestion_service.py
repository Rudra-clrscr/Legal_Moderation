"""Phase A: upload -> validate/decrypt -> OCR/extract -> chunk -> embed/store.

Validation, password checks and image conversion run before the API responds, so the client
gets immediate errors (wrong password, unsupported type). Extraction and embedding are slow and
run in a background task; the client polls the document status.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from app.errors import DocumentNotFoundError, DocumentUnreadableError, LegalBotError
from app.schemas import DocumentInfo, DocumentStatus
from app.services.chunker import MarkdownChunker
from app.services.document_parser import DocumentParser, PreparedDocument
from app.services.document_registry import DocumentRegistry
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def safe_display_name(filename: str | None) -> str:
    name = Path(filename or "document").name.strip()
    return name[:255] or "document"


class IngestionService:
    def __init__(
        self,
        registry: DocumentRegistry,
        parser: DocumentParser,
        chunker: MarkdownChunker,
        vector_store: VectorStore,
        *,
        max_concurrent: int = 1,
    ) -> None:
        self._registry = registry
        self._parser = parser
        self._chunker = chunker
        self._vs = vector_store
        self._semaphore = asyncio.Semaphore(max_concurrent)  # OCR models are memory-hungry
        self._tasks: set[asyncio.Task] = set()

    async def ingest(self, upload_path: Path, filename: str | None, password: str | None) -> DocumentInfo:
        """Takes ownership of ``upload_path``. Returns once the document is accepted for processing."""
        doc_id = self._registry.new_id()
        doc_dir = self._registry.doc_dir(doc_id, create=True)
        try:
            original = doc_dir / "original"
            shutil.move(str(upload_path), original)
            prepared = await asyncio.to_thread(self._parser.prepare, original, doc_dir, password)
        except BaseException:
            shutil.rmtree(doc_dir, ignore_errors=True)
            raise

        info = self._registry.create(doc_id, safe_display_name(filename))
        task = asyncio.create_task(self._process(doc_id, prepared))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return info

    async def _process(self, doc_id: str, prepared: PreparedDocument) -> None:
        try:
            async with self._semaphore:
                parsed = await asyncio.to_thread(self._parser.parse, prepared)
                self._registry.write_artifact(doc_id, "parsed.md", parsed.markdown)
                chunks = await asyncio.to_thread(self._chunker.chunk, doc_id, parsed.pages)
                if not chunks:
                    raise DocumentUnreadableError("No text content could be extracted from the document.")
                await self._vs.add_chunks(chunks)
            self._registry.update(
                doc_id, status=DocumentStatus.READY, chunk_count=len(chunks), parse_report=parsed.report()
            )
            logger.info("Document %s ready: %d chunks via %s", doc_id, len(chunks), parsed.parser_used)
        except DocumentNotFoundError:
            logger.info("Document %s was deleted during processing", doc_id)
            await self._discard_vectors(doc_id)
        except LegalBotError as exc:
            logger.warning("Document %s failed: %s", doc_id, exc.message)
            self._registry.mark_failed(doc_id, exc.code, exc.message)
        except Exception:
            logger.exception("Unexpected failure while processing %s", doc_id)
            self._registry.mark_failed(doc_id, "INTERNAL_ERROR", "Unexpected error while processing the document.")
        finally:
            if prepared.is_temporary:  # never leave decrypted copies on disk
                prepared.path.unlink(missing_ok=True)

    async def _discard_vectors(self, doc_id: str) -> None:
        try:
            await self._vs.delete_document(doc_id)
        except Exception:
            logger.warning("Could not remove vectors for deleted document %s", doc_id, exc_info=True)
