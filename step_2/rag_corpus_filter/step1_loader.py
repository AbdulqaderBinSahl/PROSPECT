"""
Loads the Step 2 input (a Step 1 corpus JSON or a stand-alone intent brief)
and matches PDFs to their Step 1 metadata.

Two input shapes are accepted:

* **Step 1 corpus** (``corpus.json`` / ``corpus_triaged.json``): has an
  ``intent_brief`` object and a ``papers`` list with Scopus metadata.
* **Intent brief** (``intent_brief.json``, written by Step 1 with
  ``--skip-scopus`` or by hand from ``templates/intent_brief.example.json``):
  the brief's fields at the top level, no paper metadata.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container for Step 1 output
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeAxis:
    """A single knowledge axis from Step 1."""
    name: str
    description: str
    priority: str = "secondary"  # primary | secondary | supporting


@dataclass
class PaperRecord:
    """Metadata for one paper retrieved by Step 1."""
    title: str
    doi: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None
    publication_year: Optional[int] = None
    source_title: Optional[str] = None
    citation_count: Optional[int] = None
    author_keywords: Optional[str] = None
    indexed_keywords: Optional[str] = None
    document_type: Optional[str] = None
    scopus_id: Optional[str] = None
    retrieved_by_queries: List[str] = field(default_factory=list)


@dataclass
class Step1Data:
    """Structured extraction from a Step 1 output JSON or an intent brief."""
    application_description: str
    domain: str
    knowledge_axes: List[KnowledgeAxis] = field(default_factory=list)
    scope_boundaries: List[str] = field(default_factory=list)
    content_anti_patterns: List[str] = field(default_factory=list)
    depth_expectation: Optional[str] = None
    papers: List[PaperRecord] = field(default_factory=list)
    # Convenience lookup: lowercase DOI -> PaperRecord
    _doi_index: Dict[str, PaperRecord] = field(default_factory=dict, repr=False)

    def paper_by_doi(self, doi: str) -> Optional[PaperRecord]:
        """Look up a paper record by DOI (case-insensitive)."""
        return self._doi_index.get(doi.lower().strip())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _build_step1_data(intent: dict, raw_papers: List[dict]) -> Step1Data:
    """Assemble a Step1Data from an intent-brief dict and a list of paper dicts."""
    missing = [k for k in ("domain", "application_context") if not str(intent.get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"The intent brief is missing required field(s): {', '.join(missing)}. "
            "See templates/intent_brief.example.json."
        )

    knowledge_axes = [
        KnowledgeAxis(
            name=ax["name"],
            description=ax["description"],
            priority=ax.get("priority", "secondary"),
        )
        for ax in intent.get("knowledge_axes", [])
    ]

    papers: List[PaperRecord] = []
    doi_index: Dict[str, PaperRecord] = {}

    for p in raw_papers:
        rec = PaperRecord(
            title=p.get("title", ""),
            doi=p.get("doi"),
            abstract=p.get("abstract"),
            authors=p.get("authors"),
            publication_year=p.get("publication_year"),
            source_title=p.get("source_title"),
            citation_count=p.get("citation_count"),
            author_keywords=p.get("author_keywords"),
            indexed_keywords=p.get("indexed_keywords"),
            document_type=p.get("document_type"),
            scopus_id=p.get("scopus_id"),
            retrieved_by_queries=p.get("retrieved_by_queries", []),
        )
        papers.append(rec)
        if rec.doi:
            doi_index[rec.doi.lower().strip()] = rec

    return Step1Data(
        application_description=intent.get("application_context", ""),
        domain=intent.get("domain", ""),
        knowledge_axes=knowledge_axes,
        scope_boundaries=intent.get("scope_boundaries", []),
        content_anti_patterns=intent.get("content_anti_patterns", []),
        depth_expectation=intent.get("depth_expectation"),
        papers=papers,
        _doi_index=doi_index,
    )


def load_step1_output(json_path: Path) -> Step1Data:
    """Read a Step 1 corpus JSON and extract all fields needed by Step 2.

    Extracts:
    - application_description  (from intent_brief.application_context)
    - domain                   (from intent_brief.domain)
    - knowledge_axes           (from intent_brief.knowledge_axes)
    - scope_boundaries         (from intent_brief.scope_boundaries)
    - papers with abstracts    (from papers[])
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    data = _build_step1_data(raw.get("intent_brief", {}), raw.get("papers", []))
    logger.info(
        "Loaded Step 1 metadata: domain=%r, %d knowledge axes, %d paper records "
        "(metadata only — actual PDFs may differ), depth=%r, %d content anti-patterns",
        data.domain,
        len(data.knowledge_axes),
        len(data.papers),
        data.depth_expectation,
        len(data.content_anti_patterns),
    )
    return data


def load_intent_brief(json_path: Path) -> Step1Data:
    """Read a stand-alone intent brief (no paper metadata)."""
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    data = _build_step1_data(raw, [])
    logger.info(
        "Loaded intent brief: domain=%r, %d knowledge axes, %d content anti-patterns "
        "(no paper metadata: Agent 1 will read the abstract from each PDF's first pages)",
        data.domain, len(data.knowledge_axes), len(data.content_anti_patterns),
    )
    return data


def detect_input_kind(json_path: Path) -> str:
    """Return ``"corpus"`` for a Step 1 corpus JSON, ``"intent_brief"`` for a brief."""
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "intent_brief" in raw:
        return "corpus"
    if isinstance(raw, dict) and "application_context" in raw:
        return "intent_brief"
    raise ValueError(
        f"{json_path} is neither a Step 1 corpus (with an 'intent_brief' key) nor an "
        "intent brief (with 'domain' and 'application_context'). "
        "See templates/intent_brief.example.json."
    )


def load_run_input(json_path: Path) -> Step1Data:
    """Load a Step 1 corpus JSON or an intent brief, detected from its content."""
    if detect_input_kind(json_path) == "corpus":
        return load_step1_output(json_path)
    return load_intent_brief(json_path)


# ---------------------------------------------------------------------------
# PDF discovery and matching
# ---------------------------------------------------------------------------

def _natural_sort_key(path: Path):
    """Sort key that treats numeric segments as integers (Paper2 before Paper10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def find_pdfs(papers_dir: Path) -> List[Path]:
    """Return all PDF files in the given directory (non-recursive), naturally sorted."""
    if not papers_dir.is_dir():
        logger.warning("Papers directory does not exist: %s", papers_dir)
        return []
    pdfs = sorted(papers_dir.glob("*.pdf"), key=_natural_sort_key)
    logger.info("Found %d PDFs in %s", len(pdfs), papers_dir)
    return pdfs


def _alnum(text: str) -> str:
    """Lowercase and drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_pdf_to_paper(pdf_path: Path, step1_data: Step1Data) -> Optional[PaperRecord]:
    """Match a PDF filename to a Step 1 paper record.

    A PDF matches a record when its filename contains the record's DOI (or
    Scopus ID), ignoring case and punctuation, so ``Paper12_10_1016_j_x_2020_1.pdf``
    matches DOI ``10.1016/j.x.2020.1``. When several DOIs match (one DOI can be
    a prefix of another), the longest match wins. Returns None if no record
    matches.
    """
    stem = _alnum(pdf_path.stem)
    best: Optional[PaperRecord] = None
    best_len = 0
    for paper in step1_data.papers:
        for ident in (paper.doi, paper.scopus_id):
            if not ident:
                continue
            slug = _alnum(ident)
            if slug and slug in stem and len(slug) > best_len:
                best, best_len = paper, len(slug)
    return best


def resolve_paper_context(
    pdf_path: Path,
    step1_data: Optional[Step1Data],
    first_pages_text: str,
) -> Tuple[str, str, Optional[PaperRecord]]:
    """Resolve the abstract text, domain and metadata for one paper.

    Uses the Scopus abstract when the PDF matches a Step 1 record; otherwise
    the PDF's first pages stand in for the abstract. The domain always comes
    from the intent brief. Agent 1 (A3, A3-w, A4) and the A2 baseline both
    use this, so every condition sees the same paper context.
    """
    metadata: Optional[PaperRecord] = None
    domain = ""

    if step1_data:
        domain = step1_data.domain
        metadata = match_pdf_to_paper(pdf_path, step1_data)
        if metadata and metadata.abstract:
            return metadata.abstract, domain, metadata

    return first_pages_text, domain, metadata
