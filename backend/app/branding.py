"""Product names users see in place of the underlying models.

- "BONC Astra AI" is the Qwen2.5 LLM: Ollama ``ollama_model`` (qwen2.5:3b, or qwen2.5:3b-cpu for
  resume autofill) or the ONNX fallback in ``llm_model_path``.
- "BONC Astra AI - Structural Vision Model (SVM)" is the Marker (Surya) layout + OCR chain: the
  ``svm`` and ``svm_enhanced`` parser strategies (called ``marker`` and ``marker_enhanced_ocr``
  before the rename; those names are still accepted in config and in saved records).

User-facing output (UI, /health, parse reports, warnings, errors) uses these names. Code uses
svm/SVM identifiers; the library imports (``marker``), Ollama model tag, model folders and the
README keep the real names. Both display names can be overridden in Settings.
"""

from __future__ import annotations

import re

AI_MODEL_NAME = "BONC Astra AI"
VISION_MODEL_NAME = "BONC Astra AI - Structural Vision Model (SVM)"

# Old strategy name -> current one, for .env files and saved records from before the rename.
LEGACY_PARSER_NAMES = {"marker": "svm", "marker_enhanced_ocr": "svm_enhanced"}
_SVM_PREFIXES = ("svm", "marker")
_SVM_NAME_RE = re.compile(r"\b(?:svm(?:_enhanced)?|marker(?:_enhanced_ocr)?)\b(?=:)")


def public_parser_name(name: str, vision_model_name: str = VISION_MODEL_NAME) -> str:
    """The SVM strategies (old or new names) show as the SVM; text-layer ones keep their names."""
    return vision_model_name if name.startswith(_SVM_PREFIXES) else name


def public_parser_text(text: str, vision_model_name: str = VISION_MODEL_NAME) -> str:
    """Rename "svm: RuntimeError"-style mentions in a message (also "marker:" in older records)."""
    return _SVM_NAME_RE.sub(vision_model_name, text)
