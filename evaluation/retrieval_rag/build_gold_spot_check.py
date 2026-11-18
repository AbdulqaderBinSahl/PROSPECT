"""Step 12 — A blinded spot check of the gold labels by human raters.

Draws a seeded, stratified sample of (query, paper) pairs from the bulk gold
labels, by layer and label (``gold.spot_check.allocation``, a list of
{layer, label, n}; the paper drew 52, over-sampling the 1-vs-2 boundary), and
shuffles it so raters cannot see the strata. Writes, in ``gold/spot_check/``:

  KEY.csv                   the sample with the model labels — keep it from the raters
  <rater>_workbook.xlsx     one blinded workbook per rater in ``gold.spot_check.raters``:
                            query, paper title and the same excerpt the labeller saw,
                            with empty Human_Label / Relied_On_Full_PDF / Notes columns

Then label the same pairs with the second model:
    python evaluation/retrieval_rag/label_gold.py --mode spot-check
and, when the workbooks are back, compute agreement:
    python evaluation/retrieval_rag/gold_spot_check_agreement.py

    python evaluation/retrieval_rag/build_gold_spot_check.py --settings <file>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.label_gold import gold_file, spot_key_file  # noqa: E402
from evaluation.retrieval_rag.rag import UNFILTERED, load_corpus, load_queries, rr_path  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, read_jsonl  # noqa: E402

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

INSTRUCTIONS = [
    "PURPOSE",
    "You are checking the accuracy of an automated relevance labeller, not",
    "judging any filter. For each row, decide whether the PAPER contains",
    "content that materially answers the QUERY, using only the excerpt",
    "shown (the same evidence the automated labeller saw).",
    "",
    "RUBRIC — put 0, 1 or 2 in the 'Human_Label' column:",
    "    2  Directly answers. Reports data, results, mechanisms or analysis",
    "       a domain expert would cite to answer this specific query.",
    "    1  Tangentially related. Touches the topic but does not provide",
    "       content that substantively answers the query.",
    "    0  Not relevant. Does not address the query.",
    "",
    "Be strict: in a broad corpus most papers are 0 or 1 for any specific",
    "query. Reserve 2 for papers whose content genuinely supports an answer.",
    "",
    "IF THE EXCERPT IS TRUNCATED: judge on what is shown. If you believe the",
    "full paper would change your label, still record the excerpt-based",
    "label, set 'Relied_On_Full_PDF' = y, and say why in 'Notes'.",
    "",
    "BLINDING: you are not shown the machine label or any filter result —",
    "please do not look them up. Rate independently, in order.",
]


def clean(text: str) -> str:
    return _ILLEGAL.sub(" ", text)


def draw(rows: list, allocation: list, seed: int) -> list:
    import numpy as np

    rng = np.random.default_rng(seed)
    strata: dict = {}
    for r in rows:
        strata.setdefault((int(r["layer"]), int(r["label"])), []).append(r)
    picked = []
    for a in allocation:
        key = (int(a["layer"]), int(a["label"]))
        pool = sorted(strata.get(key, []), key=lambda r: (r["query_id"], r["paper_id"]))
        idx = rng.choice(len(pool), size=min(int(a["n"]), len(pool)), replace=False)
        picked += [{"query_id": pool[i]["query_id"], "paper_id": pool[i]["paper_id"], "layer": key[0],
                    "model_label": key[1], "stratum": f"L{key[0]}_label{key[1]}"} for i in idx]
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def workbook(path: Path, sample: list, queries: dict, corpus: dict, n_chars: int) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins["A1"] = "Gold-label spot check — relevance rating"
    ins["A1"].font = Font(bold=True, size=14)
    for i, line in enumerate(INSTRUCTIONS, 3):
        ins[f"A{i}"] = line
    ins.column_dimensions["A"].width = 82

    sh = wb.create_sheet("Sample")
    cols = ["Row", "Query_ID", "Query", "Paper_ID", "Paper_Title", "Evidence_Excerpt",
            "Human_Label", "Relied_On_Full_PDF", "Notes"]
    sh.append(cols)
    for c in range(1, len(cols) + 1):
        sh.cell(row=1, column=c).font = Font(bold=True, color="FFFFFF")
        sh.cell(row=1, column=c).fill = PatternFill("solid", fgColor="1F4E78")
    for col, w in {"A": 5, "B": 10, "C": 55, "D": 26, "E": 40, "F": 90, "G": 13, "H": 11, "I": 30}.items():
        sh.column_dimensions[col].width = w
    for i, p in enumerate(sample, 1):
        paper = corpus[p["paper_id"]]
        sh.append([i, p["query_id"], clean(queries[p["query_id"]]["text"]), p["paper_id"],
                   clean(paper["title"]), clean(paper["full_text"][:n_chars]), None, None, None])
        for col in ("C", "E", "F", "I"):
            sh[f"{col}{i + 1}"].alignment = Alignment(wrap_text=True, vertical="top")
        sh.row_dimensions[i + 1].height = 140
    dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Invalid", error="Enter 0, 1, or 2")
    sh.add_data_validation(dv)
    dv.add(f"G2:G{len(sample) + 1}")
    sh.freeze_panes = "A2"
    wb.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    g = s.get("retrieval_rag.gold")
    sc = g["spot_check"]

    rows = read_jsonl(gold_file(s))
    if not rows:
        raise SystemExit("\nERROR: no bulk gold labels. Run label_gold.py first.\n")
    sample = draw(rows, sc["allocation"], int(sc.get("seed", 42)))
    with spot_key_file(s).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Row", "query_id", "paper_id", "layer", "model_label", "stratum"])
        for i, p in enumerate(sample, 1):
            w.writerow([i, p["query_id"], p["paper_id"], p["layer"], p["model_label"], p["stratum"]])
    queries = {q["query_id"]: q for q in load_queries(s, layers=(1, 3))}
    corpus = load_corpus(s, UNFILTERED)
    for rater in sc.get("raters", []) or []:
        path = rr_path(s, "gold", "spot_check", f"{rater}_workbook.xlsx")
        if path.exists():
            print(f"  {path.name} exists — kept (it may already hold ratings)")
            continue
        workbook(path, sample, queries, corpus, int(g.get("excerpt_chars", 6000)))
        print(f"  {rater}: {path}")
    print(f"Spot check: {len(sample)} pairs; key: {spot_key_file(s)} (do not share with raters)")


if __name__ == "__main__":
    main()
