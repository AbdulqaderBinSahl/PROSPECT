"""Statistics used by both evaluation parts.

Kept deliberately small and explicit so every reported number can be traced
to one function: binary classification metrics with a paired bootstrap,
agreement coefficients, Holm/Bonferroni corrections, paired Wilcoxon planned
contrasts with a rank-biserial effect size, and Wilson intervals.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── binary classification ───────────────────────────────────────────────────


def confusion(reference: Sequence[str], predicted: Sequence[str]) -> Dict[str, int]:
    """Counts with "keep" as the positive class."""
    tp = fp = fn = tn = 0
    for ref, pred in zip(reference, predicted):
        if ref == "keep" and pred == "keep":
            tp += 1
        elif ref == "discard" and pred == "keep":
            fp += 1
        elif ref == "keep" and pred == "discard":
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def classification_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def bootstrap_classification_ci(
    reference: Sequence[str], predicted: Sequence[str], n_resamples: int, seed: int,
    metrics: Iterable[str] = ("precision", "recall", "f1", "mcc"),
) -> Dict[str, Tuple[float, float]]:
    """95% percentile CIs, resampling papers with their (reference, decision) pair.

    The same resamples are used for every metric, so the result for each metric
    equals a separate bootstrap with the same seed.
    """
    ref = np.asarray(reference)
    pred = np.asarray(predicted)
    n = len(ref)
    metrics = list(metrics)
    if n == 0:
        return {m: (float("nan"), float("nan")) for m in metrics}
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_resamples) for m in metrics}
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r, p = ref[idx], pred[idx]
        cm = {
            "TP": int(np.sum((r == "keep") & (p == "keep"))),
            "FP": int(np.sum((r == "discard") & (p == "keep"))),
            "FN": int(np.sum((r == "keep") & (p == "discard"))),
            "TN": int(np.sum((r == "discard") & (p == "discard"))),
        }
        vals = classification_metrics(cm)
        for m in metrics:
            draws[m][i] = vals[m]
    return {m: (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) for m, d in draws.items()}


# ── agreement ───────────────────────────────────────────────────────────────


def fleiss_kappa_binary(votes: List[List[bool]]) -> float:
    """Fleiss' kappa for binary ratings; ``votes`` holds one list of rater votes per item."""
    from statsmodels.stats.inter_rater import fleiss_kappa

    if len(votes) < 2:
        return float("nan")
    table = np.array([[len(v) - sum(v), sum(v)] for v in votes])
    return float(fleiss_kappa(table))


def cohen_kappa(a: Sequence, b: Sequence, weights: Optional[str] = None,
                labels: Optional[List] = None) -> float:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, weights=weights, labels=labels))


def kappa_label(k: float) -> str:
    """Landis & Koch descriptive label."""
    if k != k:
        return "-"
    if k < 0:
        return "worse than chance"
    for cut, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k < cut:
            return name
    return "almost perfect"


# ── multiple comparisons ────────────────────────────────────────────────────


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = np.empty(m), 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


# ── paired contrasts ────────────────────────────────────────────────────────


def paired_bootstrap_ci(diff: np.ndarray, n_resamples: int, seed: int) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Wilcoxon effect size, -1..1)."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    r_pos, r_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total else 0.0


def planned_contrasts(
    table: pd.DataFrame, contrasts: List[Dict], n_resamples: int, seed: int,
) -> List[Dict]:
    """Paired Wilcoxon tests for a family of planned contrasts, Holm-corrected.

    ``table`` has one row per paired unit (a query) and one column per condition.
    Each contrast is ``{"name", "from", "to", "alternative"}``; the difference is
    ``to - from``. Two-sided p-values are always reported, with Holm over the
    family. A contrast whose ``alternative`` is "greater" or "less" also gets a
    directional p-value and its own Holm correction. Contrasts with fewer than
    3 pairs, or only zero differences, get p = 1 and stay in the family.
    """
    from scipy.stats import wilcoxon

    rows, p_two, p_dir = [], [], []
    for c in contrasts:
        a, b, alt = c["from"], c["to"], c.get("alternative", "two-sided")
        pair = table[[a, b]].dropna()
        diff = pair[b].to_numpy() - pair[a].to_numpy()
        row = {"contrast": f"{a}->{b}", "mechanism": c["name"], "n": len(diff),
               "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
               "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
               "alternative": alt}
        if len(diff) < 3 or np.all(diff == 0):
            pt = pd_ = 1.0
            row.update(rank_biserial=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
        else:
            pt = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox")[1])
            pd_ = pt if alt == "two-sided" else float(
                wilcoxon(diff, alternative=alt, zero_method="wilcox")[1])
            lo, hi = paired_bootstrap_ci(diff, n_resamples, seed)
            row.update(rank_biserial=rank_biserial(diff), ci_low=lo, ci_high=hi)
        row["p_two_sided"] = pt
        row["p_directional"] = pd_ if alt != "two-sided" else float("nan")
        p_two.append(pt)
        p_dir.append(pd_)
        rows.append(row)
    for r, h in zip(rows, holm(p_two)):
        r["p_two_sided_holm"] = h
    if any(c.get("alternative", "two-sided") != "two-sided" for c in contrasts):
        for r, h in zip(rows, holm(p_dir)):
            r["p_directional_holm"] = h
    return rows


def pairwise_wilcoxon(table: pd.DataFrame, conditions: List[str], n_comparisons: int) -> List[Dict]:
    """Two-sided paired Wilcoxon for every pair of conditions, Bonferroni over ``n_comparisons``."""
    from scipy.stats import wilcoxon

    rows = []
    for a, b in combinations(conditions, 2):
        pair = table[[a, b]].dropna()
        if len(pair) < 3:
            continue
        diff = pair[a].to_numpy() - pair[b].to_numpy()
        if np.all(diff == 0):
            continue
        stat, p = wilcoxon(diff, alternative="two-sided")
        rows.append({"condition_a": a, "condition_b": b, "n_pairs": len(pair),
                     "stat": float(stat), "p_raw": float(p),
                     "p_bonferroni": min(1.0, float(p) * n_comparisons)})
    return rows


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> Tuple[float, float, float]:
    """(mean, low, high) with a percentile bootstrap of the mean."""
    rng = np.random.default_rng(seed)
    if len(values) < 2:
        m = float(np.mean(values))
        return m, m, m
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_resamples)]
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, (c - m) / d), min(1.0, (c + m) / d)
