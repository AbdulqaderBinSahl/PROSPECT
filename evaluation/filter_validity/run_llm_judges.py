"""Step 5 — Every LLM judge scores every paper in the locked sample.

Appends one line per (paper, judge) to ``filter_validity/llm_scores.jsonl``
and skips pairs already there, so an interrupted run resumes. A failed call
is reported and left for the next run.

Budget guard: the run stops once the estimated cost reaches
``filter_validity.budget_usd`` (from each judge's ``cost_per_1m``). Raise the
budget and run again to continue.

    python evaluation/filter_validity/run_llm_judges.py --settings <file>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import (  # noqa: E402
    Judge,
    llm_judge_specs,
    llm_scores_file,
    paper_text,
    render_prompt,
    sample_file,
)
from evaluation.shared.settings import add_settings_arg, load_settings, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    locked = sample_file(s)
    if not locked.exists():
        raise SystemExit(f"\nERROR: locked sample not found: {locked}. Run draw_sample.py --lock.\n")
    paper_ids = json.loads(locked.read_text(encoding="utf-8"))["paper_ids"]
    budget = float(s.get("filter_validity.budget_usd", 100.0))

    out = llm_scores_file(s)
    done = {(r["paper_id"], r["judge_name"]) for r in read_jsonl(out)}
    judges = [Judge(spec) for spec in llm_judge_specs(s)]
    total = len(paper_ids) * len(judges)
    print(f"\n{len(paper_ids)} papers x {len(judges)} judges = {total} calls; "
          f"{len(done)} already done.\n")

    spent = 0.0
    with out.open("a", encoding="utf-8") as fout:
        for i, pid in enumerate(paper_ids, 1):
            try:
                prompt = render_prompt(s, paper_text(s, pid))
            except FileNotFoundError as exc:
                print(f"  [{i}/{len(paper_ids)}] SKIP {pid}: {exc}")
                continue
            for judge in judges:
                if (pid, judge.judge_id) in done:
                    continue
                if spent >= budget:
                    print(f"\nBudget guard: estimated ${spent:.2f} reached budget_usd=${budget:.2f}. "
                          "Raise it in the settings and run again to resume.")
                    return
                try:
                    r = judge.judge(pid, prompt)
                except Exception as exc:  # noqa: BLE001 — leave for the next run
                    print(f"  [{i}/{len(paper_ids)}] {judge.judge_id} FAILED on {pid}: {exc}")
                    continue
                fout.write(json.dumps(Judge.record(r), ensure_ascii=False) + "\n")
                fout.flush()
                done.add((pid, judge.judge_id))
                spent += r.cost_usd
            print(f"  [{i:>3}/{len(paper_ids)}] {pid[:45]:<45} est. cost ${spent:.2f} "
                  f"({len(done)}/{total} calls done)")

    missing = total - len(done)
    print(f"\nScores: {out}" + (f"\n{missing} calls still missing; run again to retry." if missing else ""))
    print("Next: python evaluation/filter_validity/prepare_human_packets.py")


if __name__ == "__main__":
    main()
