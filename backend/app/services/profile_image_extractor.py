"""Extract a candidate's profile headshot from a resume (PDF, Word, or image) if available."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps, UnidentifiedImageError

from app.services.document_parser import _is_docx

logger = logging.getLogger(__name__)

_MIN_PORTRAIT_SIDE = 48
_MAX_EMBEDDED_WIDTH = 900
_MAX_EMBEDDED_HEIGHT = 1100
_MIN_ASPECT_RATIO = 0.45
_MAX_ASPECT_RATIO = 2.2
_MIN_STDDEV = 8.0
_MAX_OUTPUT_DIM = 512


@lru_cache(maxsize=1)
def _load_cascades():
    """Lazily load OpenCV Haar face detectors, or return None if unavailable."""
    try:
        import cv2
        import numpy as np  # noqa: F401
    except ImportError:
        return None

    default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    alt2_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    default_cls = cv2.CascadeClassifier(default_path)
    alt2_cls = cv2.CascadeClassifier(alt2_path)
    if default_cls.empty() or alt2_cls.empty():
        logger.warning("OpenCV Haar cascades failed to load; skipping profile image detection.")
        return None
    return default_cls, alt2_cls


def _decode_pil(raw_bytes: bytes) -> Image.Image | None:
    """Decode image bytes to an opaque RGB PIL Image (compositing alpha onto white)."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as opened:
            img = ImageOps.exif_transpose(opened)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.getchannel("A"))
                return bg
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _is_candidate_portrait(img: Image.Image) -> bool:
    """Filter out icons, divider bars, flat fills, and full-page scans."""
    import numpy as np

    w, h = img.size
    if min(w, h) < _MIN_PORTRAIT_SIDE:
        return False
    if w >= _MAX_EMBEDDED_WIDTH and h >= _MAX_EMBEDDED_HEIGHT:
        return False
    ratio = w / max(1, h)
    if ratio < _MIN_ASPECT_RATIO or ratio > _MAX_ASPECT_RATIO:
        return False
    if float(np.std(np.asarray(img))) < _MIN_STDDEV:
        return False
    return True


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return (inter / max(1, union)) >= 0.3


def _detect_faces(img: Image.Image, *, strict: bool = False) -> list[tuple[int, int, int, int]]:
    """Detect human faces in an RGB image.

    When ``strict`` is True (used for full-page renders/scans), require agreement between two
    Haar cascades so dense resume text, tables, or QR codes never trigger false positives.
    """
    cascades = _load_cascades()
    if cascades is None:
        return []
    import cv2
    import numpy as np

    default_cls, alt2_cls = cascades
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    if not strict:
        faces = default_cls.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces):
            return [tuple(map(int, box)) for box in faces]
        faces_alt = alt2_cls.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        return [tuple(map(int, box)) for box in faces_alt]

    faces_def = [tuple(map(int, b)) for b in default_cls.detectMultiScale(gray, 1.1, 6, minSize=(35, 35))]
    if not faces_def:
        return []
    faces_alt = [tuple(map(int, b)) for b in alt2_cls.detectMultiScale(gray, 1.1, 5, minSize=(35, 35))]
    return [b for b in faces_def if any(_boxes_overlap(b, a) for a in faces_alt)]


def _crop_headshot(page_img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Expand a detected face box into a standard portrait crop clamped to page bounds."""
    x, y, w, h = box
    pw, ph = page_img.size
    left = max(0, int(x - 0.45 * w))
    top = max(0, int(y - 0.50 * h))
    right = min(pw, int(x + 1.45 * w))
    bottom = min(ph, int(y + 1.75 * h))
    return page_img.crop((left, top, right, bottom))


def _encode_data_url(img: Image.Image) -> str:
    w, h = img.size
    if max(w, h) > _MAX_OUTPUT_DIM:
        scale = _MAX_OUTPUT_DIM / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _iter_pdf_embedded_images(pdf_path: Path) -> Iterator[tuple[int, Image.Image]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path), strict=False)
    seen_hashes: set[str] = set()
    for page_idx, page in enumerate(reader.pages):
        try:
            images = page.images
        except Exception:
            continue
        for img_file in images:
            data = getattr(img_file, "data", None)
            if not data:
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            decoded = _decode_pil(data)
            if decoded is not None:
                yield page_idx + 1, decoded


def _iter_docx_images(docx_path: Path) -> Iterator[tuple[int, Image.Image]]:
    with zipfile.ZipFile(docx_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("word/media/"):
                continue
            decoded = _decode_pil(archive.read(name))
            if decoded is not None:
                yield 1, decoded


def _render_first_page(pdf_path: Path, dpi: int = 150) -> Image.Image | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        if len(pdf) == 0:
            return None
        return pdf[0].render(scale=dpi / 72).to_pil().convert("RGB")
    finally:
        pdf.close()


def extract_profile_image(
    src_path: Path,
    file_type: str,
    prepared_pdf_path: Path | None = None,
) -> str | None:
    """Detect and extract a candidate profile photo as a JPEG data URL, or None if unavailable."""
    if _load_cascades() is None:
        return None

    try:
        candidates: list[tuple[int, Image.Image]] = []
        has_page1_raster = False
        if file_type == "word" and _is_docx(src_path):
            candidates.extend(_iter_docx_images(src_path))
        elif file_type in {"pdf", "word"} and prepared_pdf_path is not None and prepared_pdf_path.is_file():
            for page_num, img in _iter_pdf_embedded_images(prepared_pdf_path):
                candidates.append((page_num, img))
                if page_num == 1 and min(img.size) >= _MIN_PORTRAIT_SIDE:
                    has_page1_raster = True

        best_score: tuple[int, int] | None = None
        best_img: Image.Image | None = None
        for page_num, img in candidates:
            if not _is_candidate_portrait(img):
                continue
            faces = _detect_faces(img, strict=False)
            if not faces:
                continue
            largest_face = max(w * h for _, _, w, h in faces)
            score = (1 if page_num == 1 else 0, largest_face)
            if best_score is None or score > best_score:
                best_score = score
                best_img = img

        if best_img is not None:
            return _encode_data_url(best_img)

        # Fallback for scanned PDFs, image resumes, or PDFs that embed a large raster scan/canvas
        # on page 1: skip rendering text-only PDFs that have no raster images >= 48px on page 1.
        page_img: Image.Image | None = None
        if file_type == "image":
            page_img = _decode_pil(src_path.read_bytes())
        elif (
            prepared_pdf_path is not None
            and prepared_pdf_path.is_file()
            and (file_type != "pdf" or has_page1_raster)
        ):
            page_img = _render_first_page(prepared_pdf_path)

        if page_img is not None:
            faces = _detect_faces(page_img, strict=True)
            if faces:
                chosen = max(faces, key=lambda b: b[2] * b[3])
                return _encode_data_url(_crop_headshot(page_img, chosen))
    except Exception as exc:
        logger.warning("Profile image extraction skipped for %s: %s", src_path.name, exc)

    return None
