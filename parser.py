"""
File-based resume parser.

Supports:
  - Plain text  (.txt)
  - PDF         (.pdf)  via pypdf
  - Word        (.docx) via python-docx

Returns raw extracted text; structured parsing is delegated to the LLM.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_bytes(filename: str, data: bytes) -> str:
    """
    Detect file type from *filename* extension and extract plain text.

    Parameters
    ----------
    filename : str
        Original file name (used only for extension detection).
    data : bytes
        Raw file bytes.

    Returns
    -------
    str
        Extracted plain text.  Never returns None.

    Raises
    ------
    ValueError
        If the file type is not supported.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(data)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(data)
    elif suffix in (".txt", ".text", ""):
        return data.decode("utf-8", errors="replace")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )


def extract_text_from_plain(text: str) -> str:
    """Pass-through for plain-text resume input."""
    return text.strip()


# ---------------------------------------------------------------------------
# Internal extractors
# ---------------------------------------------------------------------------

def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF parsing. "
            "Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages).strip()


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()
