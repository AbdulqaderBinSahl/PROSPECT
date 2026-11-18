"""End-to-end Step 2 runs of every condition with a fake LLM.

Checks that all five conditions run through ``run_step2.py`` in one run
folder, share one operational config (the pre-step runs once), route PDFs,
and that A2 and the multi-agent conditions see the same paper context.
"""

from __future__ import annotations

import json
import sys

import pytest

import fakes
from conftest import DOMAIN, MATCHED_ABSTRACT

CONDITIONS = ["A3w", "A3", "A4", "A2", "A1"]


@pytest.fixture
def step2(monkeypatch):
    """Import run_step2 with the model check disabled; return a runner."""
    import run_step2

    monkeypatch.setattr(run_step2, "validate_models_available", lambda *a, **k: [])
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def run(*args: str) -> None:
        monkeypatch.setattr(sys, "argv", [
            "run_step2.py", "--base-url", "http://localhost:9/v1", "--model", "fake-model", *args,
        ])
        run_step2.main()

    return run


def _decisions(run_dir, condition):
    return sorted((run_dir / condition / "paper_decisions").glob("*.json"))


def test_all_conditions_share_one_config_and_route_pdfs(run_dir, step2, monkeypatch):
    fake = fakes.install(monkeypatch)

    for condition in CONDITIONS:
        step2("--run-dir", str(run_dir), "--condition", condition)

        out = run_dir / condition
        assert len(_decisions(run_dir, condition)) == 2, condition
        assert [p.name for p in (out / "routed_pdfs" / "keep").glob("*.pdf")] == ["Paper1_10_1002_app_22271.pdf"]
        assert [p.name for p in (out / "routed_pdfs" / "discard").glob("*.pdf")] == ["Paper2_unknown_source.pdf"]
        manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["condition"] == condition
        assert manifest["inputs"]["input_path"].endswith("corpus_triaged.json")

    # The configuration pre-step ran once for the whole run folder.
    assert len(fake.labels("config_prestep")) == 1
    assert (run_dir / "operational_config.json").exists()


def test_each_condition_makes_the_expected_calls(run_dir, step2, monkeypatch):
    for condition in CONDITIONS:
        fake = fakes.install(monkeypatch)
        step2("--run-dir", str(run_dir), "--condition", condition)
        utility = fake.labels("_utility")
        weights = fake.labels("_chunk_weights")
        if condition == "A4":
            assert utility == [] and weights == []
        elif condition == "A3w":
            assert utility and len(weights) == 2
        elif condition == "A3":
            assert utility and weights == []
        elif condition == "A1":
            assert len(fake.calls) == 2  # one call per paper
        elif condition == "A2":
            assert fake.labels("a2_agg_") and all(c["model"] != "PaperProfile" for c in fake.calls)


def test_a3w_records_chunk_weights(run_dir, step2, monkeypatch):
    fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", "A3w")
    decision = json.loads(_decisions(run_dir, "A3w")[0].read_text(encoding="utf-8"))
    assert "chunk_weights" in json.dumps(decision)


def test_a2_sees_exactly_the_agent1_paper_context(run_dir, step2, monkeypatch):
    fake = fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", "A3")
    step2("--run-dir", str(run_dir), "--condition", "A2")

    for paper in ("Paper1_", "Paper2_"):
        agent1_prompt = next(
            c["prompt"] for c in fake.calls if c["label"].startswith(paper) and c["label"].endswith("_profiler")
        )
        a2_chunk_prompt = next(
            c["prompt"] for c in fake.calls if c["label"].startswith(f"a2_{paper}") and "_c0000" in c["label"]
        )
        assert agent1_prompt in a2_chunk_prompt
        assert f"APPLICATION DOMAIN:\n{DOMAIN}" in agent1_prompt

    matched_profile = next(c["prompt"] for c in fake.calls if c["label"].startswith("Paper1_") and c["label"].endswith("_profiler"))
    unmatched_profile = next(c["prompt"] for c in fake.calls if c["label"].startswith("Paper2_") and c["label"].endswith("_profiler"))
    assert MATCHED_ABSTRACT in matched_profile          # Scopus abstract for the matched PDF
    assert MATCHED_ABSTRACT not in unmatched_profile    # first pages for the unmatched PDF
    assert "unmatched0w0" in unmatched_profile


def test_rerun_without_resume_is_refused_and_resume_skips_done_papers(run_dir, step2, monkeypatch):
    fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", "A3w")
    with pytest.raises(SystemExit, match="--resume"):
        step2("--run-dir", str(run_dir), "--condition", "A3w")

    fake = fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", "A3w", "--resume")
    assert fake.calls == []  # both papers already decided
    assert len(_decisions(run_dir, "A3w")) == 2


@pytest.mark.parametrize("condition", ["A4", "A1"])
def test_failed_paper_is_reported_and_retried_on_resume(run_dir, step2, monkeypatch, capsys, condition):
    fake = fakes.install(monkeypatch)
    answer = fake._answer

    def fail_for_paper2(human_prompt, response_model, context_label):
        if "Paper2_" in context_label:
            raise RuntimeError(f"[{context_label}] All attempts exhausted")
        return answer(human_prompt, response_model, context_label)

    monkeypatch.setattr(fake, "_answer", fail_for_paper2)
    step2("--run-dir", str(run_dir), "--condition", condition)
    out = capsys.readouterr().out
    assert "1 of 2 papers have no decision" in out and "Paper2_unknown_source.pdf" in out
    assert len(_decisions(run_dir, condition)) == 1

    fakes.install(monkeypatch)  # the LLM works again
    step2("--run-dir", str(run_dir), "--condition", condition, "--resume")
    assert "have no decision" not in capsys.readouterr().out
    assert len(_decisions(run_dir, condition)) == 2
    metrics = json.loads((run_dir / condition / "run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["total_papers_failed"] == 0 and metrics["failed_papers"] == []


@pytest.mark.parametrize("condition", CONDITIONS)
def test_pdf_without_text_is_flagged_and_discarded_without_llm(run_dir, step2, monkeypatch, capsys, condition):
    from conftest import make_pdf
    from rag_corpus_filter.text_extractor import NO_TEXT_REASON

    make_pdf(run_dir / "papers" / "Paper3_scanned.pdf", words_per_page=0)
    fake = fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", condition)

    out = capsys.readouterr().out
    assert "almost no extractable text" in out and "Paper3_scanned.pdf" in out
    assert "have no decision" not in out
    assert fake.labels("Paper3_") == []  # no LLM call for it
    decision = next(p for p in _decisions(run_dir, condition) if p.name.startswith("Paper3_"))
    assert NO_TEXT_REASON in decision.read_text(encoding="utf-8")
    assert (run_dir / condition / "routed_pdfs" / "discard" / "Paper3_scanned.pdf").exists()


def test_run_metrics_share_one_format_across_conditions(run_dir, step2, monkeypatch):
    fakes.install(monkeypatch)
    shared_keys = {
        "condition", "run_timestamp", "total_papers", "total_papers_processed",
        "total_papers_failed", "failed_papers", "keep", "discard", "total_llm_calls",
        "total_wall_time_sec", "this_session_wall_time_sec", "avg_time_per_paper_sec",
        "avg_llm_calls_per_paper", "hardware_model_info", "per_paper",
    }
    for condition in CONDITIONS:
        step2("--run-dir", str(run_dir), "--condition", condition)
        out = run_dir / condition
        metrics = json.loads((out / "run_metrics.json").read_text(encoding="utf-8"))
        assert shared_keys <= set(metrics), (condition, shared_keys - set(metrics))
        assert metrics["condition"] == condition
        assert (metrics["total_papers"], metrics["keep"], metrics["discard"]) == (2, 1, 1), condition
        assert metrics["hardware_model_info"]["model"] == "fake-model"
        for row in metrics["per_paper"]:
            assert {"pdf", "decision", "total_llm_calls", "failed"} <= set(row), condition
        # Every condition keeps a copy of the config it used.
        assert (out / "operational_config.json").read_text(encoding="utf-8") == \
            (run_dir / "operational_config.json").read_text(encoding="utf-8")


def test_a_different_input_for_a_later_condition_is_refused(run_dir, step2, monkeypatch, tmp_path):
    fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--condition", "A3w")
    other = tmp_path / "brief.json"
    other.write_text(json.dumps({"domain": "x", "application_context": "y"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="same input"):
        step2("--run-dir", str(run_dir), "--condition", "A1", "--input", str(other))


def test_skip_step1_with_own_brief_and_pdfs(tmp_path, step2, monkeypatch):
    from conftest import intent_brief, make_pdf

    brief = tmp_path / "my_brief.json"
    brief.write_text(json.dumps(intent_brief()), encoding="utf-8")
    make_pdf(tmp_path / "pdfs" / "Paper1_any.pdf", tag="own")
    fake = fakes.install(monkeypatch)

    run_dir = tmp_path / "run"
    step2("--run-dir", str(run_dir), "--input", str(brief), "--papers-dir", str(tmp_path / "pdfs"))
    assert len(_decisions(run_dir, "A3w")) == 1
    profile_prompt = next(c["prompt"] for c in fake.calls if c["label"].endswith("_profiler"))
    assert f"APPLICATION DOMAIN:\n{DOMAIN}" in profile_prompt  # domain always reaches Agent 1


def test_prepare_only_writes_config_and_stops(run_dir, step2, monkeypatch):
    fake = fakes.install(monkeypatch)
    step2("--run-dir", str(run_dir), "--prepare-only")
    assert (run_dir / "operational_config.json").exists()
    assert [c["model"] for c in fake.calls] == ["OperationalConfig"]
    assert not (run_dir / "A3w").exists()
