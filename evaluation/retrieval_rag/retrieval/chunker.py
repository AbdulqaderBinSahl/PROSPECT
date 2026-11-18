"""Token-aware recursive text chunking for Step 3 indexing."""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    original_text: str
    chunk_index: int
    source_section: str
    corpus_condition: str


def chunk_paper(
    paper_id: str,
    title: str,
    full_text: str,
    corpus_condition: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Chunk]:
    """Split a paper into token-counted chunks.

    Uses ``langchain_text_splitters.RecursiveCharacterTextSplitter`` with
    tiktoken token counting when available; falls back to a built-in
    implementation otherwise.
    """
    texts = _split_text(full_text, chunk_size, chunk_overlap)

    chunks: List[Chunk] = []
    for i, raw_text in enumerate(texts):
        section = _detect_section(raw_text)

        chunks.append(
            Chunk(
                chunk_id=f"{paper_id}__chunk_{i:04d}",
                paper_id=paper_id,
                title=title,
                text=raw_text,
                original_text=raw_text,
                chunk_index=i,
                source_section=section,
                corpus_condition=corpus_condition,
            )
        )

    return chunks


# ── splitting backend ────────────────────────────────────────────────────────


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Try langchain first, fall back to built-in splitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
        )
        return splitter.split_text(text)
    except ImportError:
        logger.info(
            "langchain_text_splitters not found — using built-in splitter"
        )
        return _fallback_split(text, chunk_size, chunk_overlap)


def _fallback_split(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Simple token-aware recursive splitter using tiktoken directly."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    def tok_len(t: str) -> int:
        return len(enc.encode(t))

    def _recursive(t: str, seps: List[str]) -> List[str]:
        if tok_len(t) <= chunk_size:
            return [t] if t.strip() else []

        sep = seps[0] if seps else " "
        parts = t.split(sep)

        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if tok_len(candidate) > chunk_size and current:
                merged.append(current)
                # build overlap from tail of current
                overlap = _tail_overlap(enc, current, chunk_overlap)
                current = overlap + sep + part if overlap else part
                # if this single part is still too big, recurse with finer sep
                if tok_len(current) > chunk_size and len(seps) > 1:
                    merged.extend(_recursive(current, seps[1:]))
                    current = ""
            else:
                current = candidate

        if current.strip():
            merged.append(current)
        return merged

    return _recursive(text, SEPARATORS)


def _tail_overlap(enc, text: str, overlap_tokens: int) -> str:
    """Return the last *overlap_tokens* tokens of *text* decoded back."""
    if overlap_tokens <= 0:
        return ""
    tokens = enc.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    return enc.decode(tokens[-overlap_tokens:])


# ── section detection ────────────────────────────────────────────────────────

# Maps raw section keywords → 7 canonical labels used across all conditions.
# Order matters: first match wins on the raw header text.
_SECTION_KEYWORD_MAP: Dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "literature review": "introduction",
    "methodology": "methods",
    "methods": "methods",
    "experimental": "methods",
    "materials": "methods",
    "characterization": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "summary": "conclusion",
    # back-matter → other
    "acknowledgement": "other",
    "references": "other",
    "appendix": "other",
    "supplementary": "other",
}


# Strips leading section numbers like "3.", "2.1", "3.1.2" from a line.
_NUMBERED_PREFIX = re.compile(r'^\s*\d+(\.\d+)*\.?\s*')


def detect_section(text: str) -> str:
    """Best-effort detection of which paper section a chunk belongs to.

    Strategy (in priority order):
    1. Check each of the first 6 lines for a standalone section header,
       stripping any leading numbering (e.g. "3. Results" → "results").
    2. Fall back to scanning the first 600 characters for any keyword.

    Returns one of: abstract | introduction | methods | results |
    discussion | conclusion | other
    """
    lines = text.strip().split("\n")

    for line in lines[:6]:
        cleaned = _NUMBERED_PREFIX.sub("", line.strip()).lower()
        if not cleaned:
            continue
        for kw, canonical in _SECTION_KEYWORD_MAP.items():
            if cleaned.startswith(kw):
                return canonical

    # Broader fallback — keyword anywhere in first 600 chars
    header = text[:600].lower()
    for kw, canonical in _SECTION_KEYWORD_MAP.items():
        if kw in header:
            return canonical

    return "other"


# Keep private alias so existing internal calls still work.
_detect_section = detect_section
