"""The run folder locks its inputs so every condition sees the same ones."""

from __future__ import annotations

import json

import pytest

from conftest import intent_brief
from rag_corpus_filter import run_setup
from rag_corpus_filter.run_setup import (
    RunSetupError,
    canonical_condition,
    check_pdf_set,
    resolve_run_inputs,
)
from rag_corpus_filter.step1_loader import find_pdfs


def test_condition_names_are_normalised():
    assert canonical_condition("a3w") == "A3w"
    assert canonical_condition("A3-W") == "A3w"
    assert canonical_condition("a1") == "A1"
    with pytest.raises(RunSetupError):
        canonical_condition("A5")


def test_first_call_discovers_triaged_corpus_and_locks_it(run_dir):
    inputs = resolve_run_inputs(run_dir)
    assert inputs.input_path.name == "corpus_triaged.json"
    assert inputs.input_kind == "corpus"
    assert inputs.papers_dir == (run_dir / "papers").resolve()
    lock = json.loads((run_dir / "run_inputs.json").read_text(encoding="utf-8"))
    # paths inside the run folder are stored relative, so the folder can be moved
    assert lock["input_path"] == "step1/corpus_triaged.json"
    assert lock["papers_dir"] == "papers"


def test_later_calls_reuse_the_lock(run_dir):
    first = resolve_run_inputs(run_dir)
    second = resolve_run_inputs(run_dir)
    assert first.input_path == second.input_path and first.papers_dir == second.papers_dir


def test_a_different_input_is_refused(run_dir, tmp_path):
    resolve_run_inputs(run_dir)
    other = tmp_path / "other_brief.json"
    other.write_text(json.dumps(intent_brief()), encoding="utf-8")
    with pytest.raises(RunSetupError, match="same input"):
        resolve_run_inputs(run_dir, input_path=other)


def test_a_different_pdf_folder_is_refused(run_dir, tmp_path):
    resolve_run_inputs(run_dir)
    with pytest.raises(RunSetupError, match="same PDFs"):
        resolve_run_inputs(run_dir, papers_dir=tmp_path)


def test_editing_the_input_after_locking_is_refused(run_dir):
    inputs = resolve_run_inputs(run_dir)
    data = json.loads(inputs.input_path.read_text(encoding="utf-8"))
    data["intent_brief"]["domain"] = "something else"
    inputs.input_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RunSetupError, match="changed"):
        resolve_run_inputs(run_dir)


def test_missing_step1_output_gives_a_clear_error(tmp_path):
    with pytest.raises(RunSetupError, match="No Step 1 output"):
        resolve_run_inputs(tmp_path / "empty_run")


def test_intent_brief_input_without_step1(tmp_path):
    brief = tmp_path / "my_brief.json"
    brief.write_text(json.dumps(intent_brief()), encoding="utf-8")
    inputs = resolve_run_inputs(tmp_path / "run", input_path=brief, papers_dir=tmp_path / "pdfs")
    assert inputs.input_kind == "intent_brief"
    assert inputs.step1_data.papers == []


def test_pdf_set_is_recorded_once_then_compared(run_dir, caplog):
    inputs = resolve_run_inputs(run_dir)
    pdfs = find_pdfs(inputs.papers_dir)
    check_pdf_set(inputs, pdfs)
    recorded = json.loads(inputs.lock_path.read_text(encoding="utf-8"))["pdf_files"]
    assert len(recorded) == 2
    check_pdf_set(inputs, pdfs[:1])  # a later condition sees one PDF fewer
    assert "differs" in caplog.text


def test_operational_config_is_generated_once(run_dir, monkeypatch):
    from fakes import fake_operational_config

    calls = []

    def fake_prestep(llm, config, app_desc, **kwargs):
        calls.append(app_desc)
        return fake_operational_config()

    monkeypatch.setattr("rag_corpus_filter.config_prestep.run_config_prestep", fake_prestep)
    from rag_corpus_filter.config import PipelineConfig

    inputs = resolve_run_inputs(run_dir)
    cfg = PipelineConfig()
    cfg.paths.output_dir = run_dir
    first = run_setup.ensure_operational_config(inputs, cfg, prestep_llm=None)
    second = run_setup.ensure_operational_config(inputs, cfg, prestep_llm=None)
    assert len(calls) == 1
    assert first == second
    assert inputs.operational_config_path.exists()
