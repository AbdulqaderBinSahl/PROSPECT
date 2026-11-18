"""Step 13 — Agreement between the human raters, the labeller and the second model.

Reads the spot-check KEY, each rater's returned workbook
(``gold/spot_check/<rater>_workbook.xlsx``, column Human_Label) and the second
model's labels. If a rater's disagreements were adjudicated afterwards, put
the outcome in ``gold/spot_check/<rater>_adjudication.csv`` (columns Row,
adjudicated, reason): agreement is then reported on the adjudicated labels.

For every pair of label sources: raw agreement, Cohen's kappa (3-class),
linear-weighted kappa, and kappa on the decision that matters (label 2 vs
not). Writes ``results/gold_spot_check/`` (agreement.csv, per_row.csv, report.md).

    python evaluation/retrieval_rag/gold_spot_check_agreement.py --settings <file>
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.label_gold import spot_key_file  # noqa: E402
from evaluation.retrieval_rag.rag import results_dir, rr_path  # noqa: E402
from evaluation.shared import stats  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, read_jsonl, to_markdown  # noqa: E402


def read_workbook(path: Path) -> dict:
    """{Row: label} from a returned workbook's Sample sheet."""
    from openpyxl import load_workbook

    sheet = load_workbook(path, read_only=True, data_only=True)["Sample"]
    rows = sheet.iter_rows(values_only=True)
    header = list(next(rows))
    i_row, i_lab = header.index("Row"), header.index("Human_Label")
    out = {}
    for r in rows:
        if r[i_row] is not None and r[i_lab] not in (None, ""):
            out[int(r[i_row])] = int(r[i_lab])
    return out


def read_adjudication(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {int(r["Row"]): int(r["adjudicated"]) for r in csv.DictReader(fh)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    sc = s.get("retrieval_rag.gold.spot_check")

    with spot_key_file(s).open(encoding="utf-8") as fh:
        key = list(csv.DictReader(fh))
    sources = {"labeller": {int(r["Row"]): int(r["model_label"]) for r in key}}
    second = {(r["query_id"], r["paper_id"]): r["label"]
              for r in read_jsonl(rr_path(s, "gold", "spot_check", "second_model_labels.jsonl"))}
    if second:
        sources["second_model"] = {int(r["Row"]): int(second[(r["query_id"], r["paper_id"])])
                                   for r in key if (r["query_id"], r["paper_id"]) in second}
    for rater in sc.get("raters", []) or []:
        wb = rr_path(s, "gold", "spot_check", f"{rater}_workbook.xlsx")
        if not wb.exists():
            print(f"  {rater}: no workbook yet")
            continue
        labels = read_workbook(wb)
        adjudicated = read_adjudication(rr_path(s, "gold", "spot_check", f"{rater}_adjudication.csv"))
        if adjudicated:
            sources[f"{rater}_as_returned"] = dict(labels)
            labels.update(adjudicated)
            print(f"  {rater}: {len(adjudicated)} adjudicated labels applied")
        sources[rater] = labels

    rows = []
    for a, b in combinations(sources, 2):
        common = sorted(set(sources[a]) & set(sources[b]))
        x = [sources[a][i] for i in common]
        y = [sources[b][i] for i in common]
        if len(common) < 2:
            continue
        rows.append({"comparison": f"{a} vs {b}", "n": len(common),
                     "raw_agreement": sum(p == q for p, q in zip(x, y)) / len(common),
                     "kappa_3class": stats.cohen_kappa(x, y),
                     "kappa_linear": stats.cohen_kappa(x, y, weights="linear"),
                     "kappa_label2": stats.cohen_kappa([v == 2 for v in x], [v == 2 for v in y])})
    out = results_dir(s, "gold_spot_check")
    with (out / "agreement.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["comparison"])
        w.writeheader()
        w.writerows(rows)
    with (out / "per_row.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Row", "query_id", "paper_id", *sources])
        for r in key:
            i = int(r["Row"])
            w.writerow([i, r["query_id"], r["paper_id"], *(sources[k].get(i, "") for k in sources)])
    (out / "report.md").write_text(f"# Gold-label spot check (n = {len(key)})\n\n" + to_markdown(rows) + "\n",
                                   encoding="utf-8")
    for r in rows:
        print(f"  {r['comparison']:<35} n={r['n']}  raw={r['raw_agreement']:.3f}  "
              f"k3={r['kappa_3class']:.3f}  k_lin={r['kappa_linear']:.3f}  k_lbl2={r['kappa_label2']:.3f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
