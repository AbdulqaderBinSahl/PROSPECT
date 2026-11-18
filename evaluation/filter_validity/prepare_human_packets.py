"""Step 6 — One blinded packet per human rater.

For each rater in ``filter_validity.human_raters`` writes
``human_packets/<rater>/`` containing the sampled PDFs, ``score_sheet.csv``
(paper IDs pre-filled; score, confidence and rationale empty) and
``rubric.md`` (from ``filter_validity.human_rubric``). The packet holds no
condition names, filter decisions, LLM scores or strata.

Each rater returns the filled sheet; save it as
``filter_validity/human_scores/<rater>.csv``. An existing score sheet is never
overwritten, so re-running cannot erase a rater's work.

With no human raters configured this step is skipped, and the reference
labels come from the LLM judges alone.

    python evaluation/filter_validity/prepare_human_packets.py --settings <file>
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import (  # noqa: E402
    fv_path,
    human_raters,
    human_scores_file,
    natural_key,
    sample_file,
)
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    raters = human_raters(s)
    if not raters:
        print("No human raters configured (filter_validity.human_raters); nothing to prepare.")
        return
    paper_ids = sorted(json.loads(sample_file(s).read_text(encoding="utf-8"))["paper_ids"], key=natural_key)
    rubric = s.path("filter_validity.human_rubric").read_text(encoding="utf-8")
    app = s.application_description()

    for rater in raters:
        packet = fv_path(s, "human_packets", rater, "score_sheet.csv").parent
        missing = 0
        for pid in paper_ids:
            src, dst = s.papers_dir / f"{pid}.pdf", packet / f"{pid}.pdf"
            if dst.exists():
                continue
            if src.exists():
                shutil.copy(src, dst)
            else:
                missing += 1
                print(f"  {rater}: PDF not found for {pid}")
        sheet = packet / "score_sheet.csv"
        if not sheet.exists():
            with sheet.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["paper_id", "score", "confidence", "rationale"])
                w.writeheader()
                for pid in paper_ids:
                    w.writerow({"paper_id": pid, "score": "", "confidence": "", "rationale": ""})
        (packet / "rubric.md").write_text(
            rubric.replace("{rater_id}", rater).replace("{n_papers}", str(len(paper_ids)))
                  .replace("{application_description}", app),
            encoding="utf-8",
        )
        print(f"  {rater}: {packet}  ({len(paper_ids) - missing}/{len(paper_ids)} PDFs)")
        print(f"     return the filled sheet as {human_scores_file(s, rater)}")
    print("\nNext, once the sheets are back: python evaluation/filter_validity/build_reference_labels.py")


if __name__ == "__main__":
    main()
