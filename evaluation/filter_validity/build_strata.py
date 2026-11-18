"""Step 1 — Record which conditions kept each paper (its "decision fingerprint").

Reads ``<condition>/paper_decisions/*.json`` for every condition in the
settings and writes ``filter_validity/strata.json``:
{paper_id: {condition: kept?}}. Prints how many papers share each fingerprint
and suggests calibration papers (kept by all, kept by one, kept by none).

    python evaluation/filter_validity/build_strata.py --settings <file>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import fv_path  # noqa: E402
from evaluation.shared.decisions import load_decisions  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def fingerprint(entry: dict, conditions: list) -> str:
    kept = [c for c in conditions if entry.get(c, False)]
    return ",".join(kept) if kept else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    conditions = s.conditions

    decisions = {}
    for c in conditions:
        d = load_decisions(s.condition_dir(c))
        decisions[c] = d
        print(f"  {c:<6} {len(d):>4} papers, {sum(v == 'keep' for v in d.values()):>4} kept")

    paper_ids = sorted(set().union(*(d.keys() for d in decisions.values())))
    strata = {pid: {c: decisions[c].get(pid) == "keep" for c in conditions} for pid in paper_ids}
    out = fv_path(s, "strata.json")
    write_json(out, strata)
    print(f"\nWritten: {out} ({len(strata)} papers)")

    counts, examples = Counter(), defaultdict(list)
    for pid, entry in strata.items():
        fp = fingerprint(entry, conditions)
        counts[fp] += 1
        examples[fp].append(pid)
    print(f"\n  {'Kept by':<35} {'Papers':>6}")
    for fp, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {fp:<35} {n:>6}")

    print("\nCalibration candidates (set filter_validity.calibration_paper_ids, or leave it empty):")
    print("  kept by all:", examples.get(",".join(conditions), [])[:5])
    for c in conditions:
        print(f"  kept by {c} only:", examples.get(c, [])[:3])
    print("  kept by none:", examples.get("none", [])[:5])
    print("\nNext: python evaluation/filter_validity/extract_texts.py")


if __name__ == "__main__":
    main()
