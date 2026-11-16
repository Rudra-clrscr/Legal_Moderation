"""Wires the services together. Tests build a ``Container`` with stub services instead."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.chunker import MarkdownChunker
from app.services.document_parser import DocumentParser
from app.services.document_registry import DocumentRegistry
from app.services.ingestion_service import IngestionService
from app.services.llm_generator import LLMGenerator
from app.services.summarizer import DocumentSummarizer, SummaryJobManager
from app.services.vector_store import Embedder, VectorStore


@dataclass
class Container:
    settings: Settings
    registry: DocumentRegistry
    parser: DocumentParser
    vector_store: VectorStore
    llm: LLMGenerator
    ingestion: IngestionService
    analysis: AnalysisService
    summaries: SummaryJobManager


def build_container(settings: Settings) -> Container:
    for directory in (settings.data_dir, settings.documents_dir, settings.tmp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    registry = DocumentRegistry(settings.documents_dir)
    parser = DocumentParser.from_settings(settings)
    chunker = MarkdownChunker(
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
    )
    embedder = Embedder(settings.embedding_model, settings.embedding_device, settings.embedding_batch_size)
    vector_store = VectorStore(settings.chroma_dir, settings.chroma_collection, embedder)
    llm = LLMGenerator.from_settings(settings)

    return Container(
        settings=settings,
        registry=registry,
        parser=parser,
        vector_store=vector_store,
        llm=llm,
        ingestion=IngestionService(registry, parser, chunker, vector_store),
        analysis=AnalysisService(vector_store, llm, settings),
        summaries=SummaryJobManager(DocumentSummarizer(vector_store, llm, settings), registry),
    )
