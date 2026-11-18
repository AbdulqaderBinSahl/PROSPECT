"""Shared fixtures: import paths, synthetic PDFs, and a Step 1-style run folder.

The tests never call a real LLM or the Scopus API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "step_1", REPO_ROOT / "step_2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DOMAIN = "recycled polymer compounding"
APPLICATION = "Advisory system that suggests compound formulations from literature data."
MATCHED_DOI = "10.1002/app.22271"
MATCHED_ABSTRACT = "SCOPUS ABSTRACT: impact strength rises with elastomer content."


def make_pdf(path: Path, n_pages: int = 3, words_per_page: int = 700, tag: str = "polymer") -> Path:
    """Write a small multi-page PDF with enough words to form several chunks."""
    import fitz

    doc = fitz.open()
    for page_no in range(n_pages):
        page = doc.new_page()
        words = " ".join(f"{tag}{page_no}w{i}" for i in range(words_per_page))
        # insert_textbox wraps the text inside the page rectangle
        page.insert_textbox(page.rect + (36, 36, -36, -36), words, fontsize=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def intent_brief() -> dict:
    """A minimal valid intent brief."""
    return {
        "domain": DOMAIN,
        "application_context": APPLICATION,
        "knowledge_axes": [
            {"name": "composition-property data", "description": "Measured properties vs composition", "priority": "primary"},
        ],
        "scope_boundaries": ["medical devices"],
        "content_anti_patterns": ["Purely qualitative market commentary"],
        "depth_expectation": "deep mechanistic detail",
        "time_window": "2010-2025",
        "target_corpus_size": None,
    }


def corpus(papers: list[dict]) -> dict:
    """A Step 1 corpus JSON with the given paper records."""
    return {"intent_brief": intent_brief(), "papers": papers}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run folder with Step 1 output and two PDFs (one matches a Step 1 record)."""
    rd = tmp_path / "run"
    step1 = rd / "step1"
    step1.mkdir(parents=True)
    papers = [
        {"title": "Matched paper", "doi": MATCHED_DOI, "abstract": MATCHED_ABSTRACT,
         "source_title": "J. Appl. Polym. Sci.", "publication_year": 2005,
         "document_type": "Article", "scopus_id": "123", "retrieved_by_queries": ["q1", "q2"]},
        {"title": "Paper without a PDF", "doi": "10.1016/j.polymer.2020.1", "abstract": "Other."},
    ]
    (step1 / "corpus_triaged.json").write_text(json.dumps(corpus(papers)), encoding="utf-8")
    make_pdf(rd / "papers" / "Paper1_10_1002_app_22271.pdf", tag="matched")
    make_pdf(rd / "papers" / "Paper2_unknown_source.pdf", tag="unmatched")
    return rd
