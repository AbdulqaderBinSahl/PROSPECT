"""Step 14 — Lens B: does each corpus keep the papers that answer the queries?

From the bulk gold labels, at two thresholds — strict (label 2) and lenient
(label >= 1):

  per corpus   precision = kept gold papers / kept papers (Wilson 95% CI),
               recall = kept gold papers / all gold papers, F1, noise fraction;
               "gold papers" = the union over all layer 1 and 3 queries
  per query    recall_q = share of that query's gold papers the corpus kept
               (queries with at least one gold paper)

Planned contrasts (``statistics.contrasts``) on per-query recall: paired
Wilcoxon, Holm over the family, bootstrap CI and rank-biserial effect size;
for layers 1 and 3 pooled, and for each layer.

Writes ``results/gold/`` (corpus_metrics.csv, per_query_recall.csv,
planned_contrasts.csv, report.md).

    python evaluation/retrieval_rag/gold_metrics.py --settings <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.label_gold import gold_file  # noqa: E402
from evaluation.retrieval_rag.rag import UNFILTERED, corpora_names, load_corpus, results_dir  # noqa: E402
from evaluation.shared import stats  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, read_jsonl, to_markdown  # noqa: E402

THRESHOLDS = (("strict (label 2)", 2), ("lenient (label >= 1)", 1))


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    st = s.get("retrieval_rag.statistics")
    n_boot, seed = int(st["bootstrap"]["resamples"]), int(st["bootstrap"]["seed"])

    labels = read_jsonl(gold_file(s))
    if not labels:
        raise SystemExit("\nERROR: no gold labels. Run label_gold.py first.\n")
    corpora = {c: set(load_corpus(s, c)) for c in corpora_names(s)}
    layer_of = {r["query_id"]: int(r["layer"]) for r in labels}
    print(f"{len(labels)} labels, {len(layer_of)} queries, {len(corpora[UNFILTERED])} papers")

    corpus_rows, recall_rows, contrast_rows = [], [], []
    for name, thr in THRESHOLDS:
        gold_q: dict = {}
        for r in labels:
            gold_q.setdefault(r["query_id"], set())
            if r["label"] >= thr:
                gold_q[r["query_id"]].add(r["paper_id"])
        G = set().union(*gold_q.values())
        for c, kept in corpora.items():
            hit = len(kept & G)
            prec = hit / len(kept) if kept else float("nan")
            rec = hit / len(G) if G else float("nan")
            lo, hi = stats.wilson(hit, len(kept))
            corpus_rows.append({"threshold": name, "corpus": c, "corpus_size": len(kept), "gold_kept": hit,
                                "gold_total": len(G), "precision": prec, "precision_ci_low": lo,
                                "precision_ci_high": hi, "recall": rec, "noise_fraction": 1 - prec,
                                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
        for qid, gset in gold_q.items():
            if gset:
                recall_rows.append({"threshold": name, "query_id": qid, "layer": layer_of[qid],
                                    "n_gold": len(gset),
                                    **{c: len(corpora[c] & gset) / len(gset) for c in corpora}})
        table = pd.DataFrame([r for r in recall_rows if r["threshold"] == name])
        for scope, layers in (("layers 1+3", (1, 3)), ("layer 1", (1,)), ("layer 3", (3,))):
            sub = table[table.layer.isin(layers)].set_index("query_id")
            for row in stats.planned_contrasts(sub, st["contrasts"], n_boot, seed):
                contrast_rows.append({"threshold": name, "scope": scope, **row})

    out = results_dir(s, "gold")
    pd.DataFrame(corpus_rows).to_csv(out / "corpus_metrics.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(out / "per_query_recall.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out / "planned_contrasts.csv", index=False)
    report = ["# Lens B: gold-paper precision and recall", ""]
    for name, _ in THRESHOLDS:
        report += [f"## {name}", "", to_markdown([r for r in corpus_rows if r["threshold"] == name],
                   ["corpus", "corpus_size", "gold_kept", "gold_total", "precision", "precision_ci_low",
                    "precision_ci_high", "recall", "f1"]), "",
                   "Planned contrasts on per-query recall (difference = to - from):", "",
                   to_markdown([r for r in contrast_rows if r["threshold"] == name],
                               ["scope", "contrast", "n", "mean_diff", "ci_low", "ci_high", "rank_biserial",
                                "p_two_sided", "p_two_sided_holm"]), ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    for r in corpus_rows:
        print(f"  {r['threshold']:<22} {r['corpus']:<11} n={r['corpus_size']:>3}  "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
