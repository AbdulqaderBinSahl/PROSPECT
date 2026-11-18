"""Step 10 — Check the RAGAS judge with an independent judge on a subset.

Draws a stratified share (``self_consistency.fraction``, seeded) of the layer
1 and 3 (query, corpus) pairs, balanced across corpora, and has a judge from a
third model family (``self_consistency.judge``; the paper: Claude) re-score
the two primary metrics with direct prompts (``prompts/self_consistency/``).

Reports two agreements:
  - per pair: Pearson r between the RAGAS and the independent scores;
  - per corpus: Spearman rho between the two judges' corpus means over the
    same subset — the ranking of corpora is what the conclusions rest on.

Writes ``scores/self_consistency/*.json`` (resumable) and
``results/self_consistency/`` (agreement.json, ranking.csv, report.md).

    python evaluation/retrieval_rag/score_self_consistency.py --settings <file> [--analyse-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import (  # noqa: E402
    corpora_names,
    load_references,
    rag_output_file,
    ragas_csv,
    results_dir,
    rr_path,
)
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, to_markdown, write_json  # noqa: E402

METRICS = ["context_recall", "answer_correctness"]
FIELDS = {
    "context_recall": lambda rag, ref: dict(
        contexts="\n\n---\n\n".join(c.get("chunk_text", "") for c in rag.get("retrieved_chunks", [])),
        ground_truth=ref),
    "answer_correctness": lambda rag, ref: dict(
        question=rag.get("query_text", ""), ground_truth=ref, answer=rag.get("llm_answer", "")),
}


def select_subset(ragas, fraction: float, seed: int) -> list:
    """Stratified by corpus: int(fraction x n) pairs per corpus (at least 1), seeded."""
    import numpy as np

    df = ragas[ragas["layer"].isin([1, 3])].dropna(subset=METRICS)
    by_corpus: dict = {}
    for _, row in df.iterrows():
        by_corpus.setdefault(row["condition"], []).append((int(row["layer"]), row["query_id"], row["condition"]))
    rng = np.random.default_rng(seed)
    picked = []
    for pairs in by_corpus.values():
        n = max(1, int(len(pairs) * fraction))
        idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
        picked.extend(pairs[i] for i in idx)
    return picked


def pass_b_file(s, qid: str, corpus: str, metric: str) -> Path:
    return rr_path(s, "scores", "self_consistency", f"{qid}_{corpus}_{metric}.json")


def analyse(s, ragas, subset, judge_model: str) -> dict:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    lookup = {(r.query_id, r.condition): r for r in ragas.itertuples()}
    full = ragas[ragas["layer"].isin([1, 3])]
    corpora = corpora_names(s)
    agreement, ranking = {}, []
    for metric in METRICS:
        b = {}
        for _, qid, corpus in subset:
            f = pass_b_file(s, qid, corpus, metric)
            if f.exists():
                score = json.loads(f.read_text(encoding="utf-8")).get("score")
                if score is not None:
                    b[(qid, corpus)] = float(score)
        pairs = [(float(getattr(lookup[k], metric)), v) for k, v in b.items()
                 if k in lookup and getattr(lookup[k], metric) == getattr(lookup[k], metric)]
        r, p = pearsonr(*zip(*pairs)) if len(pairs) >= 3 else (float("nan"), float("nan"))

        def means(score_of):
            out = {}
            for c in corpora:
                vals = [v for v in (score_of(k) for k in sorted(b) if k[1] == c) if v is not None and v == v]
                out[c] = float(np.mean(vals)) if vals else float("nan")
            return out
        a_sub = means(lambda k: float(getattr(lookup[k], metric)) if k in lookup else None)
        b_sub = means(lambda k: b.get(k))
        a_full = {c: float(full[full.condition == c][metric].mean()) for c in corpora}
        both = [c for c in corpora if a_sub[c] == a_sub[c] and b_sub[c] == b_sub[c]]
        rho, p_rho = spearmanr([a_sub[c] for c in both], [b_sub[c] for c in both])
        agreement[metric] = {"pearson_r": float(r), "pearson_p": float(p), "n_pairs": len(pairs),
                             "spearman_rho_corpus_means": float(rho), "spearman_p": float(p_rho),
                             "k_corpora": len(both), "ragas_judge": s.get("retrieval_rag.ragas.judge_model"),
                             "independent_judge": judge_model}
        for c in corpora:
            ranking.append({"metric": metric, "corpus": c, "ragas_mean_subset": a_sub[c],
                            "independent_mean_subset": b_sub[c], "ragas_mean_all_L1_L3": a_full[c]})
    return {"agreement": agreement, "ranking": ranking}


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--analyse-only", action="store_true", help="No new calls; recompute the agreement")
    args = parser.parse_args()
    s = load_settings(args.settings)
    cfg = s.get("retrieval_rag.self_consistency")

    if not ragas_csv(s).exists():
        raise SystemExit("\nERROR: no RAGAS scores yet. Run score_ragas.py first.\n")
    ragas = pd.read_csv(ragas_csv(s))
    subset = select_subset(ragas, float(cfg.get("fraction", 0.25)), int(cfg.get("seed", 42)))
    print(f"Subset: {len(subset)} (query, corpus) pairs x {len(METRICS)} metrics")

    judge_model = cfg["judge"]["model"]
    if not args.analyse_only:
        client = client_for(cfg["judge"])
        prompts = {m: s.resolve(cfg["prompts"][m]).read_text(encoding="utf-8") for m in METRICS}
        refs = load_references(s)
        for layer, qid, corpus in subset:
            rag_file = rag_output_file(s, layer, qid, corpus)
            if not rag_file.exists() or qid not in refs:
                print(f"  skipped {qid} x {corpus} (no RAG output or reference)")
                continue
            rag = json.loads(rag_file.read_text(encoding="utf-8"))
            for metric in METRICS:
                out = pass_b_file(s, qid, corpus, metric)
                if out.exists():
                    continue
                try:
                    res = client.complete_json(prompts[metric].format(**FIELDS[metric](rag, refs[qid])))
                except Exception as exc:  # noqa: BLE001 — rerun to retry
                    print(f"  failed {qid} x {corpus} | {metric}: {exc}")
                    continue
                write_json(out, {**res, "query_id": qid, "condition": corpus, "metric": metric,
                                 "judge_model": client.model})
                print(f"  {qid} x {corpus} | {metric}: {res.get('score')}")

    result = analyse(s, ragas, subset, judge_model)
    out_dir = results_dir(s, "self_consistency")
    write_json(out_dir / "agreement.json", result["agreement"])
    pd.DataFrame(result["ranking"]).to_csv(out_dir / "ranking.csv", index=False)
    lines = ["# RAGAS judge vs independent judge", ""]
    for metric, a in result["agreement"].items():
        lines += [f"## {metric}", "",
                  f"- Spearman rho of corpus means: {a['spearman_rho_corpus_means']:.3f} "
                  f"(p = {a['spearman_p']:.4f}, {a['k_corpora']} corpora)",
                  f"- Pearson r per pair: {a['pearson_r']:.3f} (p = {a['pearson_p']:.4f}, n = {a['n_pairs']})", "",
                  to_markdown([r for r in result["ranking"] if r["metric"] == metric]), ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for metric, a in result["agreement"].items():
        print(f"  {metric}: rho = {a['spearman_rho_corpus_means']:.3f}, Pearson r = {a['pearson_r']:.3f} "
              f"(n = {a['n_pairs']})")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
