"""File-backed registry of uploaded documents (``<data>/documents/<doc_id>/meta.json``)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from app.errors import DocumentNotFoundError, DocumentNotReadyError
from app.schemas import DocumentInfo, DocumentStatus, ErrorInfo, utcnow

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"[0-9a-f]{32}")
_ARTIFACT_RE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
META_FILE = "meta.json"


class DocumentRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._docs: dict[str, DocumentInfo] = {}
        self._load()

    def _load(self) -> None:
        for meta in self.root.glob(f"*/{META_FILE}"):
            try:
                info = DocumentInfo.model_validate_json(meta.read_text(encoding="utf-8"))
                self._docs[info.doc_id] = info
            except Exception as exc:
                logger.warning("Skipping unreadable registry entry %s: %s", meta, exc)

    # -- Paths -------------------------------------------------------------------
    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def doc_dir(self, doc_id: str, *, create: bool = False) -> Path:
        if not _DOC_ID_RE.fullmatch(doc_id):  # also prevents path traversal
            raise DocumentNotFoundError(f"Document {doc_id!r} not found.")
        path = self.root / doc_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, info: DocumentInfo) -> None:
        path = self.doc_dir(info.doc_id, create=True) / META_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(info.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    # -- CRUD ----------------------------------------------------------------------
    def create(self, doc_id: str, filename: str) -> DocumentInfo:
        info = DocumentInfo(doc_id=doc_id, filename=filename, status=DocumentStatus.PROCESSING)
        with self._lock:
            self._write(info)
            self._docs[doc_id] = info
        return info

    def get(self, doc_id: str) -> DocumentInfo:
        self.doc_dir(doc_id)
        with self._lock:
            info = self._docs.get(doc_id)
        if info is None:
            raise DocumentNotFoundError(f"Document {doc_id!r} not found.")
        return info

    def require_ready(self, doc_id: str) -> DocumentInfo:
        info = self.get(doc_id)
        if info.status != DocumentStatus.READY:
            raise DocumentNotReadyError(
                f"Document is {info.status.value}; it must be ready before analysis.",
                details={"status": info.status.value},
            )
        return info

    def list(self) -> list[DocumentInfo]:
        with self._lock:
            return sorted(self._docs.values(), key=lambda d: d.created_at, reverse=True)

    def update(self, doc_id: str, **changes: Any) -> DocumentInfo:
        with self._lock:
            info = self.get(doc_id).model_copy(update={**changes, "updated_at": utcnow()})
            self._write(info)
            self._docs[doc_id] = info
            return info

    def mark_failed(self, doc_id: str, code: str, message: str) -> None:
        try:
            self.update(doc_id, status=DocumentStatus.FAILED, error=ErrorInfo(code=code, message=message))
        except DocumentNotFoundError:
            pass

    def delete(self, doc_id: str) -> None:
        path = self.doc_dir(doc_id)
        with self._lock:
            self.get(doc_id)
            self._docs.pop(doc_id, None)
        shutil.rmtree(path, ignore_errors=True)

    def recover_interrupted(self) -> None:
        """Documents left 'processing' by a crash/restart can never finish: mark them failed."""
        for info in self.list():
            if info.status == DocumentStatus.PROCESSING:
                self.mark_failed(info.doc_id, "INTERRUPTED", "Processing was interrupted; upload again.")

    # -- Per-document artifacts (parsed markdown, summaries) --------------------------
    def write_artifact(self, doc_id: str, name: str, content: str) -> None:
        if not _ARTIFACT_RE.fullmatch(name):
            raise ValueError(f"Invalid artifact name {name!r}")
        self.get(doc_id)
        (self.doc_dir(doc_id, create=True) / name).write_text(content, encoding="utf-8")

    def read_artifact(self, doc_id: str, name: str) -> str | None:
        if not _ARTIFACT_RE.fullmatch(name):
            raise ValueError(f"Invalid artifact name {name!r}")
        path = self.doc_dir(doc_id) / name
        return path.read_text(encoding="utf-8") if path.is_file() else None
