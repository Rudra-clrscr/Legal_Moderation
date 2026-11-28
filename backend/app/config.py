"""Application configuration, loaded from environment variables (prefix ``LEGAL_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEGAL_", env_file=".env", extra="ignore")

    # --- Storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    max_upload_mb: int = 50
    max_resume_upload_mb: int = 10
    max_pages: int = 500

    # --- Parsing / OCR -----------------------------------------------------
    # Strategies are tried in order. The first one whose output reaches
    # ``parse_quality_threshold`` wins; otherwise the best-scoring attempt is used.
    parser_strategies: list[str] = Field(
        default_factory=lambda: ["marker", "marker_enhanced_ocr", "unstructured", "pypdf"]
    )
    parse_quality_threshold: float = 0.6
    parse_min_quality: float = 0.15  # below this the document is rejected as unreadable
    min_chars_per_page: int = 200
    ocr_render_dpi: int = 200

    # --- Chunking (token counts are estimates; bge-small has a 512-token limit) ---
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 480
    chunk_min_tokens: int = 60

    # --- Embeddings & vector store -----------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    chroma_collection: str = "legal_chunks"
    retrieval_top_k: int = 3  # 3 high-relevance chunks optimizes prompt speed and avoids CPU prefill lag
    risk_queries_top_k: int = 3

    # --- LLM Providers & Inference -----------------------------------------
    # "auto" detects if local Ollama is active on port 11434 (sub-second latency & 50+ tok/s).
    # If Ollama is offline or unavailable, automatically falls back to local ONNX (DirectML / CPU).
    llm_provider: Literal["auto", "ollama", "onnx"] = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"

    # DirectML int4 model utilizes the GPU for fast token streaming; fallback to CPU folder if absent.
    llm_model_path: Path = (
        Path("models/qwen2.5-7b-instruct-awq-onnx-int4-dml")
        if Path("models/qwen2.5-7b-instruct-awq-onnx-int4-dml").exists()
        else Path("models/qwen2.5-7b-instruct-awq-onnx-int4")
    )
    llm_prompt_template: Literal["chatml", "mistral"] = "chatml"  # chatml=Qwen2.5, mistral=SaulLM
    llm_execution_provider: str | None = "dml" if Path("models/qwen2.5-7b-instruct-awq-onnx-int4-dml").exists() else None
    llm_context_window: int = 8192
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_repetition_penalty: float = 1.05
    llm_load_on_startup: bool = True

    # --- Summarization (map-reduce) ----------------------------------------
    summary_map_input_tokens: int = 2500
    summary_map_max_new_tokens: int = 450
    summary_reduce_max_new_tokens: int = 1200

    # --- Profile extraction -------------------------------------------------
    # Long resumes (6+ roles) need ~2.5k output tokens; the limit only costs time when used.
    profile_max_new_tokens: int = 3072
    # Use a PDF's own text layer for resumes when it passes the parse quality gate; scanned
    # resumes (and False here) use the same OCR chain as legal/normal documents.
    profile_prefer_text_layer: bool = True

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_resume_upload_bytes(self) -> int:
        return self.max_resume_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
