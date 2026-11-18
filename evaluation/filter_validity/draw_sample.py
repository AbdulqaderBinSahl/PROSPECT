"""Step 3 — Draw the stratified evaluation sample, then lock it.

Papers are grouped by how many conditions kept them:

    kept by all            take all
    kept by exactly one    take all  (where the conditions disagree most)
    kept by 2, 3, 4, ...   at most caps_by_n_kept[n] papers per fingerprint
    kept by none           one paper per TF-IDF k-means topic cluster

All randomness is seeded. The first run writes ``sample_preview.json``;
review it, then run again with ``--lock`` to write ``sample_locked.json``,
which every later step reads. Lock the sample before any rating starts.

    python evaluation/filter_validity/draw_sample.py --settings <file>
    python evaluation/filter_validity/draw_sample.py --settings <file> --lock
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import fv_path, sample_file, texts_dir  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402

_NAMES = {1: "singleton", 2: "pair", 3: "triple", 4: "quad"}


def classify(entry: dict, conditions: list) -> tuple[str, str]:
    """(category, fingerprint) of one paper."""
    kept = [c for c in conditions if entry.get(c, False)]
    n = len(kept)
    if n == len(conditions):
        return "consensus_keep", ",".join(kept)
    if n == 0:
        return "consensus_discard", "none"
    return _NAMES.get(n, f"kept_by_{n}"), ",".join(kept)


def topic_diverse_sample(paper_ids: list, n: int, k_clusters: int, seed: int, texts: Path) -> list:
    """One paper per TF-IDF k-means cluster (first 5,000 characters of each text)."""
    available = [pid for pid in paper_ids if (texts / f"{pid}.txt").exists()]
    if len(available) < len(paper_ids) * 0.5:
        print(f"  Warning: only {len(available)}/{len(paper_ids)} texts found; "
              "random sample for the kept-by-none group. Run extract_texts.py first.")
        return random.sample(paper_ids, min(n, len(paper_ids)))

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [(texts / f"{pid}.txt").read_text(encoding="utf-8")[:5000] for pid in available]
    n_clusters = min(k_clusters, len(available))
    X = TfidfVectorizer(max_features=500, stop_words="english").fit_transform(docs)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)

    clusters: dict = defaultdict(list)
    for pid, label in zip(available, labels):
        clusters[label].append(pid)
    sampled: list = []
    for label in range(n_clusters):
        if clusters[label]:
            sampled.append(random.choice(clusters[label]))
        if len(sampled) >= n:
            break
    remaining = [p for p in available if p not in sampled]
    if len(sampled) < n and remaining:
        sampled += random.sample(remaining, min(n - len(sampled), len(remaining)))
    return sampled[:n]


def draw(s) -> dict:
    cfg = s.get("filter_validity.sample")
    seed = int(cfg.get("seed", 42))
    caps = {int(k): int(v) for k, v in (cfg.get("caps_by_n_kept") or {}).items()}
    none_n = int(cfg.get("none_kept_sample", 10))
    clusters = int(cfg.get("none_kept_topic_clusters", 10))
    conditions = s.conditions

    strata_path = fv_path(s, "strata.json")
    if not strata_path.exists():
        raise SystemExit("\nERROR: strata.json not found. Run build_strata.py first.\n")
    strata = json.loads(strata_path.read_text(encoding="utf-8"))

    random.seed(seed)
    groups: dict = defaultdict(list)
    for pid, entry in strata.items():
        groups[classify(entry, conditions)].append(pid)

    sample, per_cat = [], defaultdict(int)
    for (cat, fp), pids in sorted(groups.items()):
        if cat == "consensus_discard":
            chosen = topic_diverse_sample(pids, none_n, clusters, seed, texts_dir(s))
        else:
            n_kept = 0 if fp == "none" else len(fp.split(","))
            cap = caps.get(n_kept) if cat not in ("consensus_keep", "singleton") else None
            chosen = pids[:] if cap is None or len(pids) <= cap else random.sample(pids, cap)
        for pid in chosen:
            sample.append({"paper_id": pid, "stratum_category": cat, "stratum_fingerprint": fp})
        per_cat[cat] += len(chosen)

    print(f"\nSample (seed={seed}):")
    print(f"  {'Group':<20} {'Sampled':>8} {'Of':>6}")
    for cat in sorted({c for c, _ in groups}, key=lambda c: (c != "consensus_keep", c)):
        avail = sum(len(v) for (c, _), v in groups.items() if c == cat)
        print(f"  {cat:<20} {per_cat[cat]:>8} {avail:>6}")
    print(f"  {'TOTAL':<20} {len(sample):>8} {len(strata):>6}")
    return {"sample": sample, "total": len(sample), "random_seed": seed,
            "generated_by": "draw_sample.py"}


def lock(s, force: bool) -> None:
    preview_path = fv_path(s, "sample_preview.json")
    if not preview_path.exists():
        raise SystemExit("\nERROR: no sample_preview.json. Run draw_sample.py without --lock first.\n")
    locked_path = sample_file(s)
    if locked_path.exists() and not force:
        raise SystemExit(f"\nERROR: {locked_path} already exists. Re-locking after rating has "
                         "started invalidates the evaluation; pass --force only if you mean it.\n")
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    locked = {
        "paper_ids": [e["paper_id"] for e in preview["sample"]],
        "stratum_assignments": {
            e["paper_id"]: {"category": e["stratum_category"], "fingerprint": e["stratum_fingerprint"]}
            for e in preview["sample"]
        },
        "total": preview["total"],
        "random_seed": preview["random_seed"],
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "script_version": "1.0.0",
    }
    write_json(locked_path, locked)
    print(f"Sample locked: {locked_path} ({locked['total']} papers, seed {locked['random_seed']}).")
    print("Commit or archive this file now, so the lock date is on record.")
    print("Next: python evaluation/filter_validity/run_calibration.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--lock", action="store_true", help="Lock the reviewed preview")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing locked sample")
    args = parser.parse_args()
    s = load_settings(args.settings)
    if args.lock:
        lock(s, args.force)
        return
    preview = draw(s)
    out = fv_path(s, "sample_preview.json")
    write_json(out, preview)
    print(f"\nPreview: {out}\nReview it, then run again with --lock.")


if __name__ == "__main__":
    main()
