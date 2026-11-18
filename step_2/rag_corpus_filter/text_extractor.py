"""
PDF text extraction with PyMuPDF and word-based chunking.

# TODO: Future version should use PyMuPDF's find_tables() to extract tables
# as structured markdown and embed them in the chunk text alongside the
# regular text extraction. This would preserve tabular data (e.g. mechanical
# property tables, composition-property matrices) that is currently flattened
# into unstructured text by get_text("text").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

from rag_corpus_filter.models import Chunk, PageText

logger = logging.getLogger(__name__)

#: PDFs averaging fewer extractable words per page than this are flagged as
#: probably scanned. Text-layer research papers have several hundred per page.
LOW_TEXT_WORDS_PER_PAGE = 50

#: Reason recorded by every condition when a PDF yields no text at all.
NO_TEXT_REASON = (
    "No extractable text in the PDF (probably a scanned image). PROSPECT does not "
    "run OCR; run OCR on the PDF first to have it assessed."
)


def words_per_page(pages: List[PageText]) -> float:
    """Average number of extractable words per page (0.0 for a PDF without pages)."""
    if not pages:
        return 0.0
    return sum(len(p.text.split()) for p in pages) / len(pages)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace, remove soft hyphens, limit consecutive newlines."""
    text = text.replace("\xad", "")          # soft hyphen
    text = re.sub(r"[ \t]+", " ", text)      # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
    return text.strip()


# ---------------------------------------------------------------------------
# Page-level extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> List[PageText]:
    """Extract text from every page of a PDF."""
    pages: List[PageText] = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        raw = doc[page_num].get_text("text")
        pages.append(PageText(page_num=page_num + 1, text=_normalise(raw)))
    doc.close()
    return pages


def extract_first_last_pages(pdf_path: Path, n_pages: int = 2) -> Tuple[str, str]:
    """Extract raw text of the first N and last N pages for Agent 1.

    Returns (first_pages_text, last_pages_text).  If the document has
    fewer than 2*n_pages pages, pages may overlap — that's fine.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)

    first_texts: List[str] = []
    for i in range(min(n_pages, total)):
        first_texts.append(_normalise(doc[i].get_text("text")))

    last_texts: List[str] = []
    start = max(total - n_pages, 0)
    for i in range(start, total):
        last_texts.append(_normalise(doc[i].get_text("text")))

    doc.close()
    return "\n\n".join(first_texts), "\n\n".join(last_texts)


# ---------------------------------------------------------------------------
# Word-based chunking with overlap
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[PageText],
    chunk_words: int = 1400,
    overlap_words: int = 200,
    min_chunk_words: int = 300,
) -> List[Chunk]:
    """Split page texts into overlapping word-based chunks.

    Small tail chunks (< min_chunk_words) are merged into the previous chunk.
    """
    # Build a flat list of (word, page_num) pairs
    word_page: List[Tuple[str, int]] = []
    for page in pages:
        words = page.text.split()
        for w in words:
            word_page.append((w, page.page_num))

    if not word_page:
        return []

    chunks: List[Chunk] = []
    idx = 0
    chunk_counter = 0

    while idx < len(word_page):
        end = min(idx + chunk_words, len(word_page))
        window = word_page[idx:end]

        text = " ".join(w for w, _ in window)
        wc = len(window)
        start_page = window[0][1]
        end_page = window[-1][1]

        # Merge small tail chunk into previous
        remaining = len(word_page) - end
        if remaining > 0 and remaining < min_chunk_words:
            tail = word_page[end:]
            text += " " + " ".join(w for w, _ in tail)
            wc += len(tail)
            end_page = tail[-1][1]
            end = len(word_page)  # consumed everything

        chunks.append(Chunk(
            chunk_id=f"c{chunk_counter:04d}",
            start_page=start_page,
            end_page=end_page,
            text=text,
            word_count=wc,
        ))
        chunk_counter += 1

        # Advance with overlap
        step = max(chunk_words - overlap_words, 1)
        if end >= len(word_page):
            break
        idx += step

    return chunks
