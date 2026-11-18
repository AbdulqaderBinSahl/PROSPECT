"""Step 15 — Lens A statistics on the RAGAS scores.

  summaries          per layer and corpus: mean (percentile bootstrap CI),
                     median, IQR, for every metric
  pairwise tests     every pair of filter conditions, per layer (1 and 3) and
                     primary metric: paired Wilcoxon (by query), Bonferroni over
                     all pairs x primary metrics of that layer
  planned contrasts  ``statistics.contrasts`` per primary metric, for layer 1,
                     layers 1+3 pooled, and layer 2 (robustness only): paired
                     Wilcoxon, Holm over the family, rank-biserial effect size,
                     bootstrap CI on the mean difference
  retention          each corpus's layer 1 context recall as % of the
                     unfiltered corpus's, with corpus size and filtering cost
                     (LLM calls from each condition's run_metrics.json)
  paraphrases        layer 1 vs layer 2 means per corpus

The references were drafted from the unfiltered corpus, so its context recall
is favoured by construction: read Lens A as retention / non-inferiority.

Writes ``results/answer_quality/``.

    python evaluation/retrieval_rag/answer_quality_stats.py --settings <file>
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import UNFILTERED, corpora_names, load_corpus, ragas_csv, results_dir  # noqa: E402
from evaluation.retrieval_rag.score_ragas import ALL_METRICS  # noqa: E402
from evaluation.shared import stats  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, to_markdown  # noqa: E402


def filtering_cost(s, condition: str) -> dict:
    p = s.condition_dir(condition) / "run_metrics.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text(encoding="utf-8"))
    calls = m.get("total_llm_calls")
    ok = (m.get("total_papers_processed") or m.get("processed") or 0)
    return {"filtering_llm_calls": calls, "calls_per_paper": round(calls / ok, 2) if calls and ok else None}


def main() -> None:
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    st = s.get("retrieval_rag.statistics")
    n_boot, seed = int(st["bootstrap"]["resamples"]), int(st["bootstrap"]["seed"])
    sb = st.get("summary_bootstrap", {"resamples": 1000, "seed": 42})
    primary = list(s.get("retrieval_rag.ragas.primary_metrics"))
    corpora = corpora_names(s)
    filters = [c for c in corpora if c != UNFILTERED]

    df = pd.read_csv(ragas_csv(s))
    out = results_dir(s, "answer_quality")

    summaries = []
    for layer in sorted(df.layer.unique()):
        for c in corpora:
            sub = df[(df.layer == layer) & (df.condition == c)]
            for m in ALL_METRICS:
                v = sub[m].dropna().to_numpy() if m in sub else np.array([])
                if len(v) == 0:
                    continue
                mean, lo, hi = stats.bootstrap_mean_ci(v, int(sb["resamples"]), int(sb["seed"]))
                summaries.append({"layer": int(layer), "corpus": c, "metric": m, "n": len(v), "mean": mean,
                                  "ci_low": lo, "ci_high": hi, "median": float(np.median(v)),
                                  "iqr": float(np.percentile(v, 75) - np.percentile(v, 25))})

    pairwise = []
    n_comp = len(list(combinations(filters, 2))) * len(primary)
    for layer in (1, 3):
        for m in primary:
            table = df[df.layer == layer].pivot_table(index="query_id", columns="condition", values=m)
            for r in stats.pairwise_wilcoxon(table, filters, n_comp):
                pairwise.append({"layer": layer, "metric": m, **r})

    contrasts = []
    for scope, layers in (("layer 1", [1]), ("layers 1+3", [1, 3]), ("layer 2 (robustness)", [2])):
        for m in primary:
            table = df[df.layer.isin(layers)].pivot_table(index="query_id", columns="condition", values=m)
            if table.empty:
                continue
            for r in stats.planned_contrasts(table, st["contrasts"], n_boot, seed):
                contrasts.append({"scope": scope, "metric": m, **r})

    l1 = df[df.layer == 1]
    base = l1[l1.condition == UNFILTERED]["context_recall"].mean()
    sizes = {c: len(load_corpus(s, c)) for c in corpora}
    retention = []
    for c in corpora:
        cr = l1[l1.condition == c]["context_recall"].mean()
        retention.append({"corpus": c, "corpus_size": sizes[c],
                          "pct_of_unfiltered": 100 * sizes[c] / sizes[UNFILTERED],
                          "context_recall_L1": cr, "recall_retention_pct": 100 * cr / base,
                          "answer_correctness_L1": l1[l1.condition == c]["answer_correctness"].mean(),
                          **(filtering_cost(s, c) if c != UNFILTERED else {})})

    paraphrase = []
    for c in corpora:
        for m in primary:
            a = df[(df.layer == 1) & (df.condition == c)][m].dropna()
            b = df[(df.layer == 2) & (df.condition == c)][m].dropna()
            if len(a) and len(b):
                paraphrase.append({"corpus": c, "metric": m, "layer1_mean": a.mean(), "layer2_mean": b.mean(),
                                   "delta": b.mean() - a.mean()})

    for name, rows in (("summaries", summaries), ("pairwise_tests", pairwise), ("planned_contrasts", contrasts),
                       ("retention", retention), ("paraphrase_robustness", paraphrase)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    report = ["# Lens A: answer quality (RAGAS)", "",
              "## Planned contrasts (difference = to - from)", "",
              to_markdown(contrasts, ["scope", "metric", "contrast", "n", "mean_diff", "ci_low", "ci_high",
                                      "rank_biserial", "p_two_sided", "p_two_sided_holm"]), "",
              "## Recall retention (layer 1)", "", to_markdown(retention), "",
              "## Pairwise tests (Bonferroni)", "", to_markdown(pairwise), "",
              "## Layer 1 vs layer 2 (paraphrases)", "", to_markdown(paraphrase), ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    for r in contrasts:
        print(f"  {r['scope']:<21} {r['metric']:<19} {r['contrast']:<9} diff={r['mean_diff']:+.4f}  "
              f"p={r['p_two_sided']:.4f}  Holm={r['p_two_sided_holm']:.4f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
