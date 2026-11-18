"""Step 4 — Calibration round: every LLM judge scores a few papers first.

Uses ``filter_validity.calibration_paper_ids`` (the paper used 2 papers kept by
all conditions, 1 kept by a single condition, 2 kept by none). If the list is
empty, those five are drawn at random (seeded) from strata.json.

Writes ``calibration/calibration_raw.jsonl`` and ``calibration_comparison.csv``.
Compare the scores side by side, and with the human raters, before the main
run. If one judge is systematically off, fix the prompt now — not after.

    python evaluation/filter_validity/run_calibration.py --settings <file>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import (  # noqa: E402
    Judge,
    fv_path,
    llm_judge_specs,
    paper_text,
    render_prompt,
)
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402


def pick_calibration_papers(s) -> list:
    ids = list(s.get("filter_validity.calibration_paper_ids", []) or [])
    if ids:
        return ids
    strata = json.loads(fv_path(s, "strata.json").read_text(encoding="utf-8"))
    n_cond = len(s.conditions)
    by_n: dict = {}
    for pid, entry in sorted(strata.items()):
        by_n.setdefault(sum(entry.values()), []).append(pid)
    rng = random.Random(int(s.get("filter_validity.sample.seed", 42)))
    picked = []
    for n_kept, k in ((n_cond, 2), (1, 1), (0, 2)):
        pool = by_n.get(n_kept, [])
        picked += rng.sample(pool, min(k, len(pool)))
    print("Calibration papers (drawn from strata.json):", picked)
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    paper_ids = pick_calibration_papers(s)
    judges = [Judge(spec) for spec in llm_judge_specs(s)]
    raw_path = fv_path(s, "calibration", "calibration_raw.jsonl")
    print(f"\nCalibration: {len(paper_ids)} papers x {len(judges)} judges\n")

    rows = []
    with raw_path.open("w", encoding="utf-8") as fout:
        for pid in paper_ids:
            print(f"  {pid}")
            try:
                prompt = render_prompt(s, paper_text(s, pid), calibration_n=len(paper_ids))
            except FileNotFoundError as exc:
                print(f"    ERROR: {exc}")
                continue
            row = {"paper_id": pid}
            for judge in judges:
                try:
                    r = judge.judge(pid, prompt)
                    fout.write(json.dumps(Judge.record(r), ensure_ascii=False) + "\n")
                    row[judge.judge_id], row[f"{judge.judge_id}_confidence"] = r.score, r.confidence
                    print(f"    {judge.judge_id}: score={r.score}, confidence={r.confidence}")
                except Exception as exc:  # noqa: BLE001 — report and continue
                    print(f"    {judge.judge_id}: ERROR — {exc}")
            rows.append(row)

    out = fv_path(s, "calibration", "calibration_comparison.csv")
    cols = ["paper_id"] + [c for j in judges for c in (j.judge_id, f"{j.judge_id}_confidence")]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nComparison: {out}")
    print("Review it (and the rationales in calibration_raw.jsonl) before run_llm_judges.py.")


if __name__ == "__main__":
    main()
