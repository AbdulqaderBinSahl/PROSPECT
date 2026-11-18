"""Step 1 orchestration in each mode, with the LLM, conversation and Scopus faked."""

from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from conftest import intent_brief


class FakeChatClient:
    """Minimal OpenAI-client stand-in for abstract triage: SKIPs titles starting 'Off-topic'."""

    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(model, messages, temperature, max_tokens):
        user = messages[-1]["content"]
        text = "SKIP | unrelated to the domain" if "Title: Off-topic" in user else "PASS"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


@pytest.fixture
def step1(monkeypatch):
    """Import run_step1 with every external dependency faked; return a runner."""
    import run_step1
    from common.env import LLMEndpoint
    from rag_corpus_builder import config
    from rag_corpus_builder.models import (
        ConceptGroup, IntentBrief, PaperMetadata, QueryResult, QueryValidationResult,
        SearchQuery, SearchStrategy,
    )

    brief = IntentBrief(**intent_brief())
    strategy = SearchStrategy(
        concept_groups=[ConceptGroup(name=f"g{i}", knowledge_axis="a", terms=["t1", "t2", "t3"]) for i in range(3)],
        queries=[SearchQuery(query_string="TITLE-ABS-KEY(x)", query_type="broad",
                             knowledge_axes_served=["a"], concept_groups_used=["g0"])],
        time_filter="PUBYEAR > 2010", subject_area_filters=["MATE"],
    )
    papers = [
        PaperMetadata(title="Relevant paper", doi="10.1002/app.22271", abstract="a", scopus_id="1"),
        PaperMetadata(title="Off-topic paper", doi="10.1/off", abstract="b", scopus_id="2"),
    ]

    monkeypatch.setattr(config, "SCOPUS_API_KEY", "test-key-1234")
    monkeypatch.setattr(run_step1, "resolve_llm_endpoint",
                        lambda *a: LLMEndpoint("http://localhost:9/v1", "k", "fake-model"))
    monkeypatch.setattr(run_step1, "OpenAI", FakeChatClient)
    monkeypatch.setattr(run_step1, "run_elicitation", lambda c, m: (brief, [{"role": "user", "content": "hi"}]))
    monkeypatch.setattr(run_step1, "generate_strategy", lambda c, m, b: strategy)
    monkeypatch.setattr(run_step1, "validate_queries", lambda c, m, s, b: (
        s, [QueryValidationResult(query_string="TITLE-ABS-KEY(x)", result_count=2, status="ok")]))
    monkeypatch.setattr(run_step1, "retrieve_papers", lambda s: (
        papers, [QueryResult(query_string="TITLE-ABS-KEY(x)", papers_retrieved=2)]))

    def run(*args: str) -> None:
        monkeypatch.setattr(sys, "argv", ["run_step1.py", *args])
        run_step1.main()

    return run


def _rows(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def test_full_step1_triages_and_writes_checklist(tmp_path, step1):
    step1("--run-dir", str(tmp_path))
    out = tmp_path / "step1"
    for name in ("intent_brief.json", "corpus.json", "corpus_triaged.json", "triage_report.json",
                 "pdf_checklist.csv", "run_log.md"):
        assert (out / name).exists(), name
    triaged = json.loads((out / "corpus_triaged.json").read_text(encoding="utf-8"))
    assert [p["title"] for p in triaged["papers"]] == ["Relevant paper"]
    assert [r["expected_filename"] for r in _rows(out / "pdf_checklist.csv")] == ["Paper1_10_1002_app_22271.pdf"]


def test_skip_triage_keeps_every_paper(tmp_path, step1):
    step1("--run-dir", str(tmp_path), "--skip-triage")
    out = tmp_path / "step1"
    assert not (out / "corpus_triaged.json").exists()
    assert len(_rows(out / "pdf_checklist.csv")) == 2


def test_skip_scopus_writes_only_the_intent_brief(tmp_path, step1, monkeypatch):
    from rag_corpus_builder import config

    monkeypatch.setattr(config, "SCOPUS_API_KEY", "")  # no key needed in this mode
    step1("--run-dir", str(tmp_path), "--skip-scopus")
    out = tmp_path / "step1"
    assert sorted(p.name for p in out.iterdir()) == ["intent_brief.json", "run_log.md"]
    assert "skipped" in (out / "run_log.md").read_text(encoding="utf-8")


def test_missing_scopus_key_fails_early(tmp_path, step1, monkeypatch):
    from rag_corpus_builder import config

    monkeypatch.setattr(config, "SCOPUS_API_KEY", "")
    with pytest.raises(SystemExit, match="SCOPUS_API_KEY"):
        step1("--run-dir", str(tmp_path))


def test_triage_only_on_an_existing_corpus(tmp_path, step1):
    step1("--run-dir", str(tmp_path), "--skip-triage")
    step1("--run-dir", str(tmp_path), "--triage-only")
    out = tmp_path / "step1"
    assert (out / "corpus_triaged.json").exists()
    assert len(_rows(out / "pdf_checklist.csv")) == 1


def test_existing_step1_output_is_not_overwritten(tmp_path, step1):
    step1("--run-dir", str(tmp_path), "--skip-triage")
    with pytest.raises(SystemExit, match="already contains"):
        step1("--run-dir", str(tmp_path))


def test_step1_output_feeds_step2(tmp_path, step1):
    """The triaged corpus is what Step 2 discovers in the run folder."""
    from rag_corpus_filter.run_setup import resolve_run_inputs

    step1("--run-dir", str(tmp_path))
    inputs = resolve_run_inputs(tmp_path)
    assert inputs.input_path.name == "corpus_triaged.json"
    assert [p.title for p in inputs.step1_data.papers] == ["Relevant paper"]
