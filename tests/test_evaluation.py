"""The evaluation code, without models or network: statistics, the rater panel
with different numbers of raters, the Part 2 statistics, prompts and settings."""

from __future__ import annotations

import csv
import json
import random
import sys

import pytest

for _dep in ("yaml", "jinja2", "scipy", "sklearn", "statsmodels", "openpyxl"):
    pytest.importorskip(_dep, reason="install evaluation/requirements.txt to run the evaluation tests")

import yaml  # noqa: E402

from conftest import REPO_ROOT, make_pdf  # noqa: E402
from evaluation.shared import stats  # noqa: E402
from evaluation.shared.llm import LLMClient, ModelSpec, parse_json_object  # noqa: E402
from evaluation.shared.settings import load_settings  # noqa: E402

CONDITIONS = ["A1", "A2", "A3", "A3w", "A4"]
EXAMPLE = REPO_ROOT / "evaluation" / "examples" / "scenario_1" / "settings.yaml"


def run(module, settings, *extra, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__file__, "--settings", str(settings), *extra])
    module.main()


# ── a synthetic PROSPECT run folder ─────────────────────────────────────────


@pytest.fixture
def eval_run(tmp_path):
    """24 papers, 5 conditions with mixed decisions, in the run-folder layout."""
    rng = random.Random(1)
    rd = tmp_path / "run"
    words = ["polymer", "blend", "impact", "recycled", "melt", "flow", "tensile", "filler"]
    pids = [f"Paper{i}_10_1000_x{i}" for i in range(1, 25)]
    for i, pid in enumerate(pids):
        make_pdf(rd / "papers" / f"{pid}.pdf", n_pages=1, words_per_page=80, tag=words[i % len(words)])
    for c in CONDITIONS:
        dec_dir = rd / c / "paper_decisions"
        dec_dir.mkdir(parents=True)
        (rd / c / "routed_pdfs" / "keep").mkdir(parents=True)
        for pid in pids:
            d = "keep" if rng.random() < 0.4 else "discard"
            if c in ("A1", "A2"):
                (dec_dir / f"{pid}.json").write_text(json.dumps({"decision": d}), encoding="utf-8")
            else:
                (dec_dir / f"{pid}_20260101_120000.json").write_text(
                    json.dumps({"decision": d, "run_metrics": {"pdf": f"{pid}.pdf"}}), encoding="utf-8")
            if d == "keep":
                (rd / c / "routed_pdfs" / "keep" / f"{pid}.pdf").write_bytes(
                    (rd / "papers" / f"{pid}.pdf").read_bytes())
        (rd / c / "run_metrics.json").write_text(
            json.dumps({"total_llm_calls": 100, "total_papers_processed": 24}), encoding="utf-8")
    (rd / "operational_config.json").write_text(json.dumps({
        "application_description": "Advisory system for recycled polymer blends.",
        "representative_queries": [f"Question {i}?" for i in range(1, 5)]}), encoding="utf-8")
    return rd, pids


def write_settings(tmp_path, rd, humans):
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data.update(run_dir=str(rd), workspace=str(tmp_path / "ws"), application_description_file=None)
    fv = data["filter_validity"]
    fv.update(sample_file=None, calibration_paper_ids=[], human_raters=humans,
              human_rubric="evaluation/filter_validity/human_rubric_template.md")
    fv["sample"].update(none_kept_sample=2, none_kept_topic_clusters=2)
    fv["bootstrap"] = {"resamples": 200, "seed": 42}
    rr = data["retrieval_rag"]
    rr["queries"]["dir"] = None
    rr["references"]["dir"] = None
    rr["statistics"]["bootstrap"] = {"resamples": 200, "seed": 42}
    rr["gold"]["spot_check"]["allocation"] = [{"layer": 1, "label": 2, "n": 3}, {"layer": 1, "label": 0, "n": 3}]
    p = tmp_path / "eval_settings.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ── statistics ──────────────────────────────────────────────────────────────


def test_holm_and_classification_metrics():
    assert stats.holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    cm = stats.confusion(["keep", "keep", "discard", "discard"], ["keep", "discard", "keep", "discard"])
    assert cm == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
    m = stats.classification_metrics(cm)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5 and m["mcc"] == 0.0
    lo, hi = stats.wilson(30, 40)
    assert lo < 0.75 < hi


def test_planned_contrasts_keep_untestable_contrasts_in_the_holm_family():
    import pandas as pd

    table = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "B": [0.3, 0.5, 0.4, 0.8, 0.9, 0.7],
                          "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    rows = stats.planned_contrasts(table, [{"name": "x", "from": "A", "to": "B"},
                                           {"name": "same", "from": "A", "to": "C"}], 200, 42)
    assert rows[1]["p_two_sided"] == 1.0                    # all-zero differences
    assert rows[0]["p_two_sided_holm"] == pytest.approx(min(1.0, 2 * rows[0]["p_two_sided"]))
    assert rows[0]["mean_diff"] > 0 and rows[0]["rank_biserial"] == 1.0


# ── Part 1: the panel works with any number of human raters ────────────────


@pytest.mark.parametrize("humans", [[], ["H1", "H2"], ["H1", "H2", "H3"]])
def test_filter_validity_pipeline(tmp_path, eval_run, monkeypatch, humans):
    from evaluation.filter_validity import (
        build_reference_labels,
        build_strata,
        compute_metrics,
        draw_sample,
        extract_texts,
        prepare_human_packets,
    )

    rd, pids = eval_run
    st = write_settings(tmp_path, rd, humans)
    for m in (build_strata, extract_texts, draw_sample):
        run(m, st, monkeypatch=monkeypatch)
    run(draw_sample, st, "--lock", monkeypatch=monkeypatch)
    fv = tmp_path / "ws" / "filter_validity"
    sample = json.loads((fv / "sample_locked.json").read_text(encoding="utf-8"))["paper_ids"]
    assert sample and set(sample) <= set(pids)
    with pytest.raises(SystemExit, match="already exists"):
        run(draw_sample, st, "--lock", monkeypatch=monkeypatch)

    # Ratings: three LLM judges and the configured humans.
    rng = random.Random(7)
    with (fv / "llm_scores.jsonl").open("w", encoding="utf-8") as fh:
        for pid in sample:
            for j in ("llm-a", "llm-b", "llm-c"):
                fh.write(json.dumps({"paper_id": pid, "judge_name": j, "score": rng.randint(1, 4),
                                     "confidence": rng.choice(["low", "high"]), "rationale": "r"}) + "\n")
    run(prepare_human_packets, st, monkeypatch=monkeypatch)
    for h in humans:
        packet = fv / "human_packets" / h
        assert (packet / "score_sheet.csv").exists() and "{rater_id}" not in (packet / "rubric.md").read_text(encoding="utf-8")
        (fv / "human_scores").mkdir(exist_ok=True)
        with (fv / "human_scores" / f"{h}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["paper_id", "score", "confidence", "rationale"])
            for pid in sample:
                w.writerow([pid, rng.randint(1, 4), "high", "r"])

    run(build_reference_labels, st, monkeypatch=monkeypatch)
    labels = json.loads((fv / "reference_labels.json").read_text(encoding="utf-8"))
    panel = 3 + len(humans)
    needed = panel // 2 + 1
    for v in labels.values():
        assert v["n_raters"] == panel
        votes = sum(e["score"] >= 3 for e in v["scores"].values())
        assert v["binary_label"] == ("keep" if votes >= needed else "discard")
        assert v["controversy_flag"] == (needed <= votes < panel)

    run(compute_metrics, st, monkeypatch=monkeypatch)
    out = fv / "metrics"
    with (out / "condition_metrics.csv").open(encoding="utf-8") as fh:
        assert [r["condition"] for r in csv.DictReader(fh)] == CONDITIONS
    refs = {r["reference"] for r in csv.DictReader((out / "alternative_references.csv").open(encoding="utf-8"))}
    assert any(r.startswith("LLM majority") for r in refs)
    assert any(r.startswith("all human raters") for r in refs) == (len(humans) >= 2)
    assert all(f"{h} alone" in refs for h in humans)


# ── Part 2: statistics on synthetic scores ──────────────────────────────────


def test_rag_statistics(tmp_path, eval_run, monkeypatch):
    import pandas as pd

    from evaluation.retrieval_rag import answer_quality_stats, build_corpora, build_gold_spot_check, gold_metrics

    rd, pids = eval_run
    st = write_settings(tmp_path, rd, ["H1"])
    run(build_corpora, st, monkeypatch=monkeypatch)
    rr = tmp_path / "ws" / "retrieval_rag"
    corpora = {c: json.loads((rr / "corpora" / f"{c}_corpus.json").read_text(encoding="utf-8"))
               for c in ["unfiltered", *CONDITIONS]}
    assert set(corpora["unfiltered"]) == set(pids)

    rng = random.Random(3)
    rows = [{"query_id": f"Q{q:03d}", "layer": layer, "condition": c, "context_recall": rng.random(),
             "answer_correctness": rng.random(), "faithfulness": rng.random(),
             "context_precision": rng.random(), "answer_relevancy": rng.random()}
            for layer, qs in ((1, range(1, 9)), (3, range(20, 24))) for q in qs for c in corpora]
    (rr / "scores").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(rr / "scores" / "ragas_scores.csv", index=False)
    run(answer_quality_stats, st, monkeypatch=monkeypatch)
    con = pd.read_csv(rr / "results" / "answer_quality" / "planned_contrasts.csv")
    assert set(con.scope) == {"layer 1", "layers 1+3"} and set(con.contrast) == {"A1->A2", "A2->A3", "A3->A3w", "A3->A4"}
    assert (con.p_two_sided_holm >= con.p_two_sided).all()

    (rr / "gold").mkdir()
    with (rr / "gold" / "gold_labels.jsonl").open("w", encoding="utf-8") as fh:
        for q in range(1, 7):
            for pid in pids:
                fh.write(json.dumps({"query_id": f"Q{q:03d}", "layer": 1, "paper_id": pid,
                                     "label": rng.choice([0, 0, 1, 2])}) + "\n")
    run(gold_metrics, st, monkeypatch=monkeypatch)
    cm = pd.read_csv(rr / "results" / "gold" / "corpus_metrics.csv")
    unf = cm[(cm.corpus == "unfiltered")]
    assert (unf.recall == 1.0).all()

    (rr / "queries").mkdir(exist_ok=True)
    (rr / "queries" / "layer_1.json").write_text(json.dumps(
        [{"query_id": f"Q{q:03d}", "text": f"Question {q}?", "layer": 1} for q in range(1, 7)]), encoding="utf-8")
    run(build_gold_spot_check, st, monkeypatch=monkeypatch)
    key = list(csv.DictReader((rr / "gold" / "spot_check" / "KEY.csv").open(encoding="utf-8")))
    assert len(key) == 6 and sorted(int(r["model_label"]) for r in key) == [0, 0, 0, 2, 2, 2]
    assert (rr / "gold" / "spot_check" / "H1_workbook.xlsx").exists()


# ── prompts, settings and model specs ───────────────────────────────────────


def test_example_settings_render_complete_prompts():
    from evaluation.filter_validity.panel import render_prompt
    from evaluation.retrieval_rag.rag import generator_prompt

    s = load_settings(str(EXAMPLE))
    judge = render_prompt(s, "PAPER TEXT")
    assert "rPP/automotive formulation" in judge and "{{" not in judge and "PAPER TEXT" in judge
    assert render_prompt(s, "X", calibration_n=5).startswith("# CALIBRATION ROUND — 5 papers only")
    gen = generator_prompt(s, "QUESTION", "EVIDENCE")
    assert gen.startswith("You are an expert polymer engineering advisor") and "{" not in gen


def test_template_settings_warn_about_todo(capsys):
    load_settings(str(REPO_ROOT / "evaluation" / "settings.yaml"))
    assert "TODO values" in capsys.readouterr().out


def test_model_specs_and_json_parsing(monkeypatch):
    with pytest.raises(ValueError, match="Unknown"):
        ModelSpec.from_settings({"provider": "openai", "model": "m", "temprature": 0})
    with pytest.raises(ValueError, match="provider"):
        ModelSpec.from_settings({"provider": "nope", "model": "m"})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        LLMClient(ModelSpec(provider="openai", model="m"))
    assert parse_json_object('```json\n{"score": 3}\n```') == {"score": 3}
    assert parse_json_object('Here: {"label": 2, "rationale": "x"} done') == {"label": 2, "rationale": "x"}


def test_judge_reply_validation():
    from evaluation.filter_validity.panel import parse_judgement

    assert parse_judgement('{"score": 4, "confidence": "High", "rationale": "data"}', "p")["confidence"] == "high"
    for bad in ('{"score": 5, "confidence": "high", "rationale": "x"}',
                '{"score": 3, "confidence": "sure", "rationale": "x"}',
                '{"score": 3, "confidence": "low", "rationale": ""}'):
        with pytest.raises(ValueError):
            parse_judgement(bad, "p")
