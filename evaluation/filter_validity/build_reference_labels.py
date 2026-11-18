"""Step 7 — Combine all raters into one reference label per paper.

A rater votes "keep" when its score is at least ``consensus.keep_score``
(default 3 on the 1-4 scale). A paper's reference label is "keep" when at
least ``consensus.min_keep_votes`` raters vote keep ("majority" = more than
half of the configured panel; the paper used 3 of 5). Fix this rule before
any rating starts.

A paper is flagged "controversial" when it is labelled keep without a
unanimous vote (for 5 raters and 3 votes: 3 or 4 keep votes).

Writes ``filter_validity/reference_labels.json``.

    python evaluation/filter_validity/build_reference_labels.py --settings <file>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import (  # noqa: E402
    all_raters,
    all_scores,
    fv_path,
    keep_score,
    min_keep_votes,
    sample_file,
)
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def reference_label(scores: dict, keep_at: int, votes_needed: int) -> tuple[str, int, bool]:
    """(label, keep votes, controversial) for one paper's {rater: {score, ...}}."""
    votes = sum(1 for e in scores.values() if isinstance(e.get("score"), int) and e["score"] >= keep_at)
    n = sum(1 for e in scores.values() if e.get("score") is not None)
    label = "keep" if votes >= votes_needed else "discard"
    return label, votes, votes_needed <= votes < n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    paper_ids = json.loads(sample_file(s).read_text(encoding="utf-8"))["paper_ids"]
    raters = all_raters(s)
    keep_at, needed = keep_score(s), min_keep_votes(s)
    scores = all_scores(s)
    print(f"\nPanel: {', '.join(raters)}. Keep = score >= {keep_at} from >= {needed} of {len(raters)} raters.\n")

    labels, incomplete = {}, []
    for pid in paper_ids:
        entry = {r: scores[pid][r] for r in raters if r in scores.get(pid, {})}
        if not entry:
            continue
        if len(entry) < len(raters):
            incomplete.append((pid, sorted(set(raters) - set(entry))))
        label, votes, controversial = reference_label(entry, keep_at, needed)
        labels[pid] = {"scores": entry, "n_raters": len(entry), "keep_votes": votes,
                       "binary_label": label, "controversy_flag": controversial}

    out = fv_path(s, "reference_labels.json")
    write_json(out, labels)
    keep = sum(v["binary_label"] == "keep" for v in labels.values())
    print(f"Reference labels: {out}")
    print(f"  labelled {len(labels)}/{len(paper_ids)}: keep {keep}, discard {len(labels) - keep}, "
          f"controversial {sum(v['controversy_flag'] for v in labels.values())}")
    if incomplete:
        print(f"  Warning: {len(incomplete)} papers lack some raters, e.g. {incomplete[:3]}. "
              "Finish the ratings before reporting.")
    print("Next: python evaluation/filter_validity/compute_metrics.py")


if __name__ == "__main__":
    main()
