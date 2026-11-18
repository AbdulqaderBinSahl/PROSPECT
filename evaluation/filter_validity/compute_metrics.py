"""Step 8 — Score each condition against the reference, and check the raters.

Writes to ``filter_validity/metrics/``:

  condition_metrics.csv     per condition: TP/FP/FN/TN, precision, recall, F1,
                            MCC, each with a 95% percentile bootstrap CI
                            (papers resampled with their reference label)
  agreement.csv             Fleiss' kappa (binary keep/discard) for the LLM
                            judges, all raters and the humans; mean pairwise
                            quadratic-weighted kappa on the 1-4 scores; the same
                            on the high-confidence subset
  rater_pairs.csv           every rater pair: Cohen's kappa (binary), quadratic-
                            weighted kappa, Spearman rho, % binary agreement
  score_distributions.csv   each rater's 1-4 score counts and keep rate
  alternative_references.csv  F1 of every condition, and its rank, when the
                            reference is replaced by: an LLM majority, all LLMs
                            agreeing, all human raters agreeing, or each rater alone
  disagreements.csv         every (paper, condition) where the decision differs
                            from the reference, with the raters' rationales
  report.md                 all of the above, readable

    python evaluation/filter_validity/compute_metrics.py --settings <file>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import (  # noqa: E402
    fv_path,
    human_raters,
    keep_score,
    llm_judge_specs,
    min_keep_votes,
    sample_file,
)
from evaluation.shared import stats  # noqa: E402
from evaluation.shared.decisions import load_decisions  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, to_markdown  # noqa: E402


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def votes(labels: dict, pids: list, raters: list, keep_at: int) -> list:
    return [[labels[p]["scores"][r]["score"] >= keep_at for r in raters] for p in pids]


def condition_metrics(s, labels, decisions, n_boot, seed) -> list:
    pids = list(labels)
    ref = [labels[p]["binary_label"] for p in pids]
    rows = []
    for cond, dec in decisions.items():
        pred = [dec.get(p, "discard") for p in pids]
        cm = stats.confusion(ref, pred)
        m = stats.classification_metrics(cm)
        ci = stats.bootstrap_classification_ci(ref, pred, n_boot, seed)
        row = {"condition": cond, **cm, **m}
        for k, (lo, hi) in ci.items():
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
        rows.append(row)
    return rows


def agreement(labels, pids, llms, humans, keep_at, min_high) -> tuple[list, list]:
    """Group-level and pairwise agreement on papers rated by everyone."""
    everyone = llms + humans
    groups = [("LLM judges", llms), ("all raters", everyone)]
    if len(humans) >= 2:
        groups.append(("human raters", humans))

    def mean_weighted(raters, papers):
        vals = [stats.cohen_kappa([labels[p]["scores"][a]["score"] for p in papers],
                                  [labels[p]["scores"][b]["score"] for p in papers],
                                  weights="quadratic", labels=[1, 2, 3, 4])
                for a, b in combinations(raters, 2)]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    high = [p for p in pids
            if sum(e.get("confidence") == "high" for e in labels[p]["scores"].values()) >= min_high]
    group_rows = []
    for subset_name, papers in (("all sampled papers", pids), (f">= {min_high} raters 'high'", high)):
        for name, raters in groups:
            if len(raters) < 2 or len(papers) < 2:
                continue
            fk = stats.fleiss_kappa_binary(votes(labels, papers, raters, keep_at))
            wk = mean_weighted(raters, papers)
            group_rows.append({"papers": subset_name, "group": name, "raters": " ".join(raters),
                               "n_papers": len(papers), "fleiss_kappa_binary": fk,
                               "fleiss_label": stats.kappa_label(fk),
                               "mean_weighted_kappa_quadratic": wk})

    from scipy.stats import spearmanr

    pair_rows = []
    for subset_name, papers in (("all sampled papers", pids), (f">= {min_high} raters 'high'", high)):
        if len(papers) < 2:
            continue
        for a, b in combinations(everyone, 2):
            sa = [labels[p]["scores"][a]["score"] for p in papers]
            sb = [labels[p]["scores"][b]["score"] for p in papers]
            ba, bb = [x >= keep_at for x in sa], [x >= keep_at for x in sb]
            rho, p_rho = spearmanr(sa, sb) if len(papers) >= 3 else (float("nan"), float("nan"))
            pair_rows.append({
                "papers": subset_name, "rater_a": a, "rater_b": b, "n_papers": len(papers),
                "cohen_kappa_binary": stats.cohen_kappa(ba, bb),
                "weighted_kappa_quadratic": stats.cohen_kappa(sa, sb, weights="quadratic", labels=[1, 2, 3, 4]),
                "spearman_rho": float(rho), "spearman_p": float(p_rho),
                "binary_agreement_pct": 100 * sum(x == y for x, y in zip(ba, bb)) / len(papers),
            })
    return group_rows, pair_rows


def rank(values: dict) -> dict:
    """1 = best; ties share the better rank (1, 1, 3)."""
    ordered = sorted(values.items(), key=lambda x: -x[1])
    ranks, prev, prev_rank = {}, None, 0
    for i, (k, v) in enumerate(ordered, 1):
        if v == prev:
            ranks[k] = prev_rank
        else:
            ranks[k], prev_rank, prev = i, i, v
    return ranks


def alternative_references(labels, pids, decisions, llms, humans, keep_at, votes_needed) -> list:
    n_llm = len(llms)
    refs = [(f"panel rule (>= {votes_needed} of {len(llms) + len(humans)})",
             {p: labels[p]["binary_label"] for p in pids})]
    if n_llm >= 2:
        k = n_llm // 2 + 1
        refs.append((f"LLM majority (>= {k} of {n_llm})",
                     {p: "keep" if sum(labels[p]["scores"][j]["score"] >= keep_at for j in llms) >= k
                      else "discard" for p in pids}))
        refs.append((f"all LLMs ({n_llm} of {n_llm})",
                     {p: "keep" if all(labels[p]["scores"][j]["score"] >= keep_at for j in llms)
                      else "discard" for p in pids}))
    if len(humans) >= 2:
        refs.append((f"all human raters ({len(humans)} of {len(humans)})",
                     {p: "keep" if all(labels[p]["scores"][h]["score"] >= keep_at for h in humans)
                      else "discard" for p in pids}))
    for r in llms + humans:
        refs.append((f"{r} alone", {p: "keep" if labels[p]["scores"][r]["score"] >= keep_at
                                    else "discard" for p in pids}))
    rows = []
    for name, ref in refs:
        f1 = {}
        for cond, dec in decisions.items():
            cm = stats.confusion([ref[p] for p in pids], [dec.get(p, "discard") for p in pids])
            m = stats.classification_metrics(cm)
            f1[cond] = m["f1"]
            rows.append({"reference": name, "n_keep": sum(v == "keep" for v in ref.values()),
                         "condition": cond, **cm, **m})
        ranks = rank(f1)
        for row in rows[-len(decisions):]:
            row["f1_rank"] = ranks[row["condition"]]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    locked = json.loads(sample_file(s).read_text(encoding="utf-8"))
    sample = set(locked["paper_ids"])
    labels_path = fv_path(s, "reference_labels.json")
    if not labels_path.exists():
        raise SystemExit("\nERROR: reference_labels.json not found. Run build_reference_labels.py.\n")
    labels = {p: v for p, v in json.loads(labels_path.read_text(encoding="utf-8")).items() if p in sample}
    decisions = {c: {p: d for p, d in load_decisions(s.condition_dir(c)).items() if p in sample}
                 for c in s.conditions}

    llms = [j["id"] for j in llm_judge_specs(s)]
    humans = [h for h in human_raters(s) if any(h in v["scores"] for v in labels.values())]
    everyone = llms + humans
    complete = [p for p in labels if all(r in labels[p]["scores"] for r in everyone)]
    keep_at, needed = keep_score(s), min_keep_votes(s)
    boot = s.get("filter_validity.bootstrap", {}) or {}
    n_boot, seed = int(boot.get("resamples", 10_000)), int(boot.get("seed", 42))
    min_high = int(s.get("filter_validity.high_confidence_min_raters", max(2, len(everyone) - 1)))
    print(f"\n{len(labels)} labelled papers ({len(complete)} rated by all of {', '.join(everyone)}), "
          f"{len(decisions)} conditions.\n")

    cm_rows = condition_metrics(s, labels, decisions, n_boot, seed)
    groups, pairs = agreement(labels, complete, llms, humans, keep_at, min_high)
    dist = []
    for r in everyone:
        sc = [labels[p]["scores"][r]["score"] for p in complete]
        dist.append({"rater": r, "n": len(sc), "mean": sum(sc) / len(sc) if sc else float("nan"),
                     **{f"score_{k}": sc.count(k) for k in (1, 2, 3, 4)},
                     "keep_pct": 100 * sum(x >= keep_at for x in sc) / len(sc) if sc else float("nan")})
    alt = alternative_references(labels, complete, decisions, llms, humans, keep_at, needed)

    disagreements = []
    for p, lab in labels.items():
        for cond, dec in decisions.items():
            pred = dec.get(p, "discard")
            if pred != lab["binary_label"]:
                disagreements.append({
                    "paper_id": p, "condition": cond, "decision": pred,
                    "reference": lab["binary_label"], "error": "FP" if pred == "keep" else "FN",
                    "keep_votes": lab["keep_votes"],
                    "rationales": " | ".join(f"{r}: {e.get('rationale', '')[:200]}"
                                             for r, e in lab["scores"].items()),
                })

    out = fv_path(s, "metrics", "report.md").parent
    write_csv(out / "condition_metrics.csv", cm_rows)
    write_csv(out / "agreement.csv", groups)
    write_csv(out / "rater_pairs.csv", pairs)
    write_csv(out / "score_distributions.csv", dist)
    write_csv(out / "alternative_references.csv", alt)
    write_csv(out / "disagreements.csv", disagreements)

    for r in cm_rows:
        print(f"  {r['condition']:<6} TP={r['TP']:>3} FP={r['FP']:>3} FN={r['FN']:>3} TN={r['TN']:>3}  "
              f"F1={r['f1']:.3f} [{r['f1_ci_low']:.3f}, {r['f1_ci_high']:.3f}]  MCC={r['mcc']:.3f}")
    for g in groups:
        print(f"  Fleiss kappa, {g['group']} ({g['papers']}): {g['fleiss_kappa_binary']:.3f}")

    firsts = {c: sum(1 for r in alt if r["condition"] == c and r["f1_rank"] == 1) for c in decisions}
    n_refs = len({r["reference"] for r in alt})
    report = [
        "# Filter validity: results", "",
        f"- Sampled papers: {locked.get('total')} (seed {locked.get('random_seed')}, "
        f"locked {locked.get('locked_at')})",
        f"- Labelled: {len(labels)}; rated by all raters: {len(complete)}",
        f"- Raters: {', '.join(everyone)}",
        f"- Reference rule: keep if score >= {keep_at} from >= {needed} of {len(llms) + len(human_raters(s))} raters",
        f"- Reference keep / discard: {sum(v['binary_label'] == 'keep' for v in labels.values())} / "
        f"{sum(v['binary_label'] == 'discard' for v in labels.values())}; controversial "
        f"{sum(v['controversy_flag'] for v in labels.values())}",
        f"- Bootstrap: {n_boot} resamples, seed {seed}", "",
        "## Each condition against the reference", "",
        to_markdown(cm_rows, ["condition", "TP", "FP", "FN", "TN", "precision", "precision_ci_low",
                              "precision_ci_high", "recall", "recall_ci_low", "recall_ci_high", "f1",
                              "f1_ci_low", "f1_ci_high", "mcc", "mcc_ci_low", "mcc_ci_high"]), "",
        "## Rater agreement", "", to_markdown(groups), "",
        to_markdown(pairs), "",
        "## Score distributions", "", to_markdown(dist), "",
        "## F1 under alternative references", "",
        to_markdown(alt, ["reference", "n_keep", "condition", "precision", "recall", "f1", "f1_rank"]), "",
        "Times ranked first: " + ", ".join(f"{c} {n}/{n_refs}" for c, n in firsts.items()), "",
        f"Disagreements: {len(disagreements)} (paper, condition) pairs; see disagreements.csv.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
