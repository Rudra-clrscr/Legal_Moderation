"""Embedding + vector storage (ChromaDB, local and persistent).

Heavy imports (``chromadb``, ``sentence_transformers``) are deferred so the rest of the app,
and the test suite, can run without them.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.services.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """``BAAI/bge-small-en-v1.5`` via sentence-transformers. Embeddings are L2-normalised."""

    # BGE v1.5 retrieval instruction: applied to queries only, never to passages.
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            return self._model

    def warm_up(self) -> None:
        self._get_model()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        return self.embed_documents([self.QUERY_INSTRUCTION + q for q in queries])


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float | None = None  # cosine similarity, higher is better


class VectorStore:
    _UPSERT_BATCH = 256

    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # -- Writes --------------------------------------------------------------
    async def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        await asyncio.to_thread(self._add_chunks, list(chunks))

    def _add_chunks(self, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), self._UPSERT_BATCH):
            batch = chunks[start : start + self._UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
                embeddings=self._embedder.embed_documents([c.embedding_text for c in batch]),
            )

    async def delete_document(self, doc_id: str) -> None:
        await asyncio.to_thread(self._collection.delete, where={"doc_id": doc_id})

    # -- Reads ---------------------------------------------------------------
    async def search(self, doc_id: str, query: str, top_k: int) -> list[RetrievedChunk]:
        results = await self.search_many(doc_id, [query], top_k)
        return results[0]

    async def search_many(
        self, doc_id: str, queries: Sequence[str], top_k: int
    ) -> list[list[RetrievedChunk]]:
        """One ranked result list per query (results are not merged)."""
        return await asyncio.to_thread(self._search_many, doc_id, list(queries), top_k)

    def _search_many(self, doc_id: str, queries: list[str], top_k: int) -> list[list[RetrievedChunk]]:
        if not queries:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed_queries(queries),
            n_results=top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"],
        )
        ranked: list[list[RetrievedChunk]] = []
        for ids, docs, metas, dists in zip(
            result["ids"], result["documents"], result["metadatas"], result["distances"]
        ):
            ranked.append(
                [
                    RetrievedChunk(Chunk.from_record(i, d or "", m or {}), score=1.0 - float(dist))
                    for i, d, m, dist in zip(ids, docs, metas, dists)
                ]
            )
        return ranked

    async def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks of a document in reading order (used by map-reduce summarisation)."""
        result = await asyncio.to_thread(
            self._collection.get, where={"doc_id": doc_id}, include=["documents", "metadatas"]
        )
        chunks = [
            Chunk.from_record(i, d or "", m or {})
            for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])
        ]
        return sorted(chunks, key=lambda c: c.index)
