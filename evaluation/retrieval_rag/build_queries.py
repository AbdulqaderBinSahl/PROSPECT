"""Step 3 — Layer 1 queries: the operational config's representative queries, verbatim.

Writes ``queries/layer_1.json`` with IDs Q001, Q002, ... and a timestamp.
Optionally label each query with a knowledge axis
(``retrieval_rag.queries.knowledge_axes``, one label per query, in order).
Fix the query set before any RAG run.

    python evaluation/retrieval_rag/build_queries.py --settings <file> [--force]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import queries_dir  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing layer_1.json")
    args = parser.parse_args()
    s = load_settings(args.settings)

    out = queries_dir(s) / "layer_1.json"
    if out.exists() and not args.force:
        raise SystemExit(f"\nERROR: {out} exists. Pass --force to rebuild it (only before any RAG run).\n")
    queries = s.operational_config().get("representative_queries", [])
    if not queries:
        raise SystemExit("\nERROR: the operational config has no representative_queries.\n")
    axes = list(s.get("retrieval_rag.queries.knowledge_axes", []) or [])
    now = datetime.now(timezone.utc).isoformat()
    layer = [{"query_id": f"Q{i + 1:03d}", "text": text,
              "source": f"operational_config.representative_queries[{i}]",
              "knowledge_axis": axes[i] if i < len(axes) else "unassigned",
              "authored_before": now, "layer": 1}
             for i, text in enumerate(queries)]
    if len({q["text"] for q in layer}) != len(layer):
        print("Warning: duplicate query texts.")
    write_json(out, layer)
    print(f"Layer 1: {len(layer)} queries -> {out}")
    print("Next: write_held_out_queries.py (layer 3) and generate_paraphrases.py (layer 2)")


if __name__ == "__main__":
    main()
