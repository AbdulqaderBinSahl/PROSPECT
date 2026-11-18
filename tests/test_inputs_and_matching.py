"""Step 2 input loading, PDF-to-metadata matching, and the Step 1 PDF checklist."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conftest import APPLICATION, DOMAIN, corpus, intent_brief
from rag_corpus_builder.output_handler import expected_pdf_filename, write_pdf_checklist
from rag_corpus_filter.step1_loader import (
    detect_input_kind,
    load_run_input,
    match_pdf_to_paper,
    resolve_paper_context,
)


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_corpus_and_intent_brief_load_the_same_application(tmp_path):
    c = _write(tmp_path / "corpus.json", corpus([{"title": "t", "doi": "10.1/x", "abstract": "a"}]))
    b = _write(tmp_path / "intent_brief.json", intent_brief())

    assert detect_input_kind(c) == "corpus"
    assert detect_input_kind(b) == "intent_brief"
    from_corpus, from_brief = load_run_input(c), load_run_input(b)
    for data in (from_corpus, from_brief):
        assert data.domain == DOMAIN
        assert data.application_description == APPLICATION
        assert data.content_anti_patterns == ["Purely qualitative market commentary"]
    assert len(from_corpus.papers) == 1 and from_brief.papers == []


def test_intent_brief_requires_domain_and_application(tmp_path):
    brief = intent_brief()
    brief["domain"] = ""
    with pytest.raises(ValueError, match="domain"):
        load_run_input(_write(tmp_path / "b.json", brief))


def test_unrecognised_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        detect_input_kind(_write(tmp_path / "x.json", {"foo": 1}))


# DOIs with punctuation that the old matcher (which only ignored / . - _) missed.
TRICKY_DOIS = [
    "10.1002/app.22271",
    "10.1016/S0141-3910(03)00091-0",
    "10.1002/1097-4628(20000815)77:7<1591::AID-APP20>3.0.CO;2-M",
    "10.4271/2000-01-1127",
]


@pytest.mark.parametrize("doi", TRICKY_DOIS)
def test_checklist_filename_matches_back_to_its_record(doi):
    data = load_run_input_from_records([{"title": "t", "doi": doi, "abstract": "a"}])
    fname = expected_pdf_filename(7, doi, None)
    assert match_pdf_to_paper(Path(fname), data).doi == doi


def test_longest_doi_wins_when_one_doi_prefixes_another():
    short, long_ = "10.1016/j.x.2020.1", "10.1016/j.x.2020.10"
    data = load_run_input_from_records([
        {"title": "short", "doi": short}, {"title": "long", "doi": long_},
    ])
    assert match_pdf_to_paper(Path(expected_pdf_filename(1, long_, None)), data).doi == long_
    assert match_pdf_to_paper(Path(expected_pdf_filename(1, short, None)), data).doi == short


def test_scopus_id_match_and_no_match():
    data = load_run_input_from_records([{"title": "t", "doi": None, "scopus_id": "85012345678"}])
    assert match_pdf_to_paper(Path("Paper3_scopus_85012345678.pdf"), data) is not None
    assert match_pdf_to_paper(Path("Paper4_something_else.pdf"), data) is None


def test_paper_context_uses_scopus_abstract_or_first_pages():
    data = load_run_input_from_records([{"title": "t", "doi": "10.1/abc", "abstract": "SCOPUS"}])
    assert resolve_paper_context(Path("P_10_1_abc.pdf"), data, "FIRST PAGES")[0] == "SCOPUS"
    abstract, domain, meta = resolve_paper_context(Path("P_other.pdf"), data, "FIRST PAGES")
    assert (abstract, domain, meta) == ("FIRST PAGES", DOMAIN, None)


def test_pdf_checklist_lists_every_paper(tmp_path):
    c = _write(tmp_path / "corpus.json", corpus([
        {"title": "A", "doi": "10.1/a"},
        {"title": "B", "doi": None, "scopus_id": "999"},
        {"title": "C"},
    ]))
    rows = list(csv.DictReader((write_pdf_checklist(c, tmp_path)).open(encoding="utf-8")))
    assert [r["expected_filename"] for r in rows] == [
        "Paper1_10_1_a.pdf", "Paper2_scopus_999.pdf", "Paper3_NO_IDENTIFIER.pdf",
    ]
    assert rows[0]["doi_link"] == "https://doi.org/10.1/a"
    assert "cannot match" in rows[2]["note"]


def load_run_input_from_records(records):
    """Build Step1Data from paper records without touching disk."""
    from rag_corpus_filter.step1_loader import _build_step1_data

    return _build_step1_data(intent_brief(), records)
