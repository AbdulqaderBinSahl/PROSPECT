"""
Backfill missing abstracts in an existing Step 1 corpus JSON via the
Scopus Abstract Retrieval API. The file is updated in place.

Usage (from the repository root):
    python step_1/tools/backfill_abstracts.py runs/my_topic/step1/corpus.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.env import load_env, scopus_keys  # noqa: E402

load_env()
API_KEY, INST_TOKEN = scopus_keys()
ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/"
REQUEST_DELAY = 0.5  # seconds between API calls


def fetch_abstract(scopus_id: str) -> str | None:
    """Fetch abstract for a single paper via the Abstract Retrieval API."""
    headers = {
        "X-ELS-APIKey": API_KEY,
        "Accept": "application/json",
    }
    if INST_TOKEN:
        headers["X-ELS-Insttoken"] = INST_TOKEN

    url = f"{ABSTRACT_URL}{scopus_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        core = (
            data
            .get("abstracts-retrieval-response", {})
            .get("coredata", {})
        )
        abstract = core.get("dc:description")
        if isinstance(abstract, dict):
            abstract = abstract.get("$")
        return abstract.strip() if abstract and isinstance(abstract, str) else None
    except Exception as exc:
        print(f"    WARN: Failed for scopus_id {scopus_id}: {exc}")
        return None


def main():
    if not API_KEY:
        print("ERROR: SCOPUS_API_KEY not set.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python step_1/tools/backfill_abstracts.py <corpus_json_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    papers = corpus.get("papers", [])
    missing = [(i, p) for i, p in enumerate(papers) if not p.get("abstract") and p.get("scopus_id")]

    print(f"Total papers: {len(papers)}")
    print(f"Missing abstracts (with scopus_id): {len(missing)}")

    if not missing:
        print("Nothing to backfill.")
        return

    filled = 0
    for count, (i, paper) in enumerate(missing, 1):
        abstract = fetch_abstract(paper["scopus_id"])
        if abstract:
            papers[i]["abstract"] = abstract
            filled += 1

        if count % 50 == 0 or count == len(missing):
            print(f"  Progress: {count}/{len(missing)} checked, {filled} abstracts fetched")

        time.sleep(REQUEST_DELAY)

    print(f"\nBackfill complete: {filled}/{len(missing)} abstracts fetched.")

    # Save back to the same file
    print(f"Saving to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
