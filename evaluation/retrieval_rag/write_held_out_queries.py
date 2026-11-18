"""Step 5 — Layer 3: held-out queries written by a person.

Held-out queries test whether a filtered corpus also serves questions the
filter never saw. Each must be within the application's scope and must not
restate a layer 1 query; you confirm both for every query. Aim for about 10.
Saved to ``queries/layer_3.json`` as Q_L3_01, Q_L3_02, ... (incrementally, so
you can stop and resume).

Interactive:     python evaluation/retrieval_rag/write_held_out_queries.py --settings <file>
From a file:     ... --from-file my_queries.txt --author <name>   (one query per line)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import load_layer, queries_dir  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def yes(prompt: str) -> bool:
    while True:
        a = input(f"  {prompt} [y/n]: ").strip().lower()
        if a in ("y", "yes", "n", "no"):
            return a.startswith("y")


def entry(index: int, text: str, author: str) -> dict:
    return {"query_id": f"Q_L3_{index:02d}", "text": text, "source": f"held_out_authored_by_{author}",
            "knowledge_axis": "held_out", "authored_before": datetime.now(timezone.utc).isoformat(),
            "author": author, "layer": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--from-file", help="Text file with one query per line")
    parser.add_argument("--author", default="", help="Author name or initials")
    args = parser.parse_args()
    s = load_settings(args.settings)

    out = queries_dir(s) / "layer_3.json"
    layer3 = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    if args.from_file:
        lines = [x.strip() for x in Path(args.from_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        for text in lines:
            layer3.append(entry(len(layer3) + 1, text, args.author or "unknown"))
        write_json(out, layer3)
        print(f"Layer 3: {len(layer3)} queries -> {out}")
        return

    print("\nApplication:\n  " + s.application_description()[:600])
    print("\nLayer 1 (do not restate these):")
    for q in load_layer(s, 1):
        print(f"  {q['query_id']}: {q['text'][:100]}")
    author = args.author or input("\nYour name or initials: ").strip() or "unknown"
    print("Type a query, or 'list', 'undo', 'done'.")
    while True:
        try:
            raw = input(f"\n[{len(layer3)} accepted] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if raw.lower() == "done":
            break
        if raw.lower() == "list":
            for q in layer3:
                print(f"  {q['query_id']}: {q['text']}")
            continue
        if raw.lower() == "undo":
            if layer3:
                print(f"  Removed {layer3.pop()['query_id']}")
                write_json(out, layer3)
            continue
        if not raw:
            continue
        if not yes("Within the application's scope?") or not yes("Not a restatement of a layer 1 query?"):
            print("  Discarded.")
            continue
        layer3.append(entry(len(layer3) + 1, raw, author))
        write_json(out, layer3)
        print(f"  Accepted as {layer3[-1]['query_id']}.")
    print(f"\nLayer 3: {len(layer3)} queries -> {out}")
    if len(layer3) < 8:
        print("Warning: fewer than 8 held-out queries; consider adding more.")


if __name__ == "__main__":
    main()
