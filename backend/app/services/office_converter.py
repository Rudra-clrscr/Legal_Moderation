"""Word (.doc / .docx) -> PDF conversion, so Word files go through the same extraction chain.

Backends, tried in order:
  1. LibreOffice (``soffice --headless``), cross-platform, the usual choice for servers
  2. Microsoft Word through COM automation (Windows with Word installed, needs ``pywin32``)

Converting to PDF (instead of reading the Word XML directly) keeps page numbers in citations
consistent with what the user sees when they open the document.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from app.errors import DocumentUnreadableError, InvalidPasswordError, LegalBotError, PasswordRequiredError

logger = logging.getLogger(__name__)

_SOFFICE_CANDIDATES = (
    "soffice",
    "libreoffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)
# Word prompts for a password when it's opened without one. Passing a password that can't be
# right makes it fail instead of hanging an invisible dialog. Unprotected files ignore it.
_NO_PASSWORD_SENTINEL = "\u0001no-password\u0001"
_WD_FORMAT_PDF = 17
_WD_DO_NOT_SAVE = 0


def find_soffice() -> str | None:
    for candidate in _SOFFICE_CANDIDATES:
        path = shutil.which(candidate) or (candidate if os.path.isfile(candidate) else None)
        if path:
            return path
    return None


def word_com_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application"))
        import win32com.client  # noqa: F401
    except (OSError, ImportError):
        return False
    return True


class OfficeConverter:
    def __init__(self, timeout_s: int = 180) -> None:
        self.timeout_s = timeout_s

    def available_backends(self) -> list[str]:
        backends = []
        if find_soffice():
            backends.append("libreoffice")
        if word_com_available():
            backends.append("word")
        return backends

    def to_pdf(self, src: Path, dst: Path, password: str | None = None) -> None:
        backends = self.available_backends()
        if not backends:
            raise LegalBotError(
                "Word documents need LibreOffice or Microsoft Word on the server to be converted."
            )
        errors: list[str] = []
        for backend in backends:
            try:
                if backend == "libreoffice":
                    if password:  # soffice has no reliable CLI password option; let Word try
                        errors.append("libreoffice: password-protected files not supported")
                        continue
                    self._libreoffice(src, dst)
                else:
                    self._word(src, dst, password)
                if dst.is_file() and dst.stat().st_size > 0:
                    return
                errors.append(f"{backend}: produced no output")
            except (PasswordRequiredError, InvalidPasswordError):
                raise
            except Exception as exc:
                logger.warning("Word conversion via %s failed: %s", backend, exc)
                errors.append(f"{backend}: {type(exc).__name__}")
        raise DocumentUnreadableError(f"The Word document could not be converted ({'; '.join(errors)}).")

    def _libreoffice(self, src: Path, dst: Path) -> None:
        soffice = find_soffice()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            # A private profile lets conversions run while the user has LibreOffice open.
            profile = Path(tmp, "profile").as_uri()
            # soffice names its output after the input file, so give the copy a .doc/.docx name.
            staged = Path(tmp, "input" + (src.suffix or ".doc"))
            shutil.copyfile(src, staged)
            subprocess.run(
                [soffice, f"-env:UserInstallation={profile}", "--headless", "--convert-to", "pdf",
                 "--outdir", tmp, str(staged)],
                check=True, timeout=self.timeout_s, capture_output=True,
            )
            shutil.move(str(Path(tmp, "input.pdf")), dst)

    def _word(self, src: Path, dst: Path, password: str | None) -> None:
        import pythoncom
        import pywintypes
        import win32com.client

        pythoncom.CoInitialize()  # runs in a worker thread
        word = None
        doc = None
        try:
            word = win32com.client.DispatchEx("Word.Application")  # private instance
            word.Visible = False
            word.DisplayAlerts = 0
            try:
                doc = word.Documents.Open(
                    str(src.resolve()), ConfirmConversions=False, ReadOnly=True,
                    AddToRecentFiles=False, PasswordDocument=password or _NO_PASSWORD_SENTINEL,
                    Visible=False, OpenAndRepair=True, NoEncodingDialog=True,
                )
            except pywintypes.com_error as exc:
                if "password" in str(exc).lower():
                    if password:
                        raise InvalidPasswordError("The password supplied for this document is incorrect.") from exc
                    raise PasswordRequiredError("This document is password-protected. Provide its password.") from exc
                raise
            doc.ExportAsFixedFormat(str(dst.resolve()), _WD_FORMAT_PDF)
        finally:
            if doc is not None:
                doc.Close(_WD_DO_NOT_SAVE)
            if word is not None:
                word.Quit(_WD_DO_NOT_SAVE)
            pythoncom.CoUninitialize()
