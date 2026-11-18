"""Step 4 — Layer 2: paraphrase each layer 1 query, with a person reviewing each one.

The paraphrase model (``retrieval_rag.queries.paraphrase.model``) writes
``per_query`` paraphrases per query. You then accept, edit, pick or reject
them. Accepted ones go to ``queries/layer_2.json`` with IDs like Q001_P01.

Layer 2 tests robustness to wording only: a paraphrase shares its parent's
reference answer and gold papers, so it is never pooled with layers 1 and 3.

    python evaluation/retrieval_rag/generate_paraphrases.py --settings <file> [--resume] [--accept-all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import load_layer, queries_dir  # noqa: E402
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def review(qid: str, original: str, options: list) -> list:
    """Interactive accept / edit / select / reject."""
    print(f"\n{qid}: {original}")
    for i, p in enumerate(options, 1):
        print(f"  [{i}] {p}")
    print("  Enter = accept all | r = reject | 1,3 = keep these | e2 = edit #2")
    accepted = list(options)
    while True:
        choice = input("Choice: ").strip()
        if choice == "":
            return accepted
        if choice.lower() == "r":
            return []
        if choice.startswith("e") and choice[1:].isdigit():
            i = int(choice[1:]) - 1
            if 0 <= i < len(accepted):
                new = input(f"  New text for [{i + 1}]: ").strip()
                if new:
                    accepted[i] = new
            continue
        try:
            keep = [options[int(x) - 1] for x in choice.split(",") if 0 < int(x) <= len(options)]
            if keep:
                return keep
        except ValueError:
            pass
        print("  Not understood; try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--resume", action="store_true", help="Skip queries already paraphrased")
    parser.add_argument("--accept-all", action="store_true", help="No interactive review (not recommended)")
    args = parser.parse_args()
    s = load_settings(args.settings)
    cfg = s.get("retrieval_rag.queries.paraphrase")
    n = int(cfg.get("per_query", 1))
    template = s.resolve(cfg["prompt"]).read_text(encoding="utf-8")
    client = client_for(cfg["model"])

    out = queries_dir(s) / "layer_2.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if (args.resume and out.exists()) else []
    done = {q["source_query_id"] for q in existing}
    items = list(existing)
    for q in load_layer(s, 1):
        if q["query_id"] in done:
            continue
        prompt = template.format(query=q["text"], n=n, domain=cfg["domain"], expert=cfg["expert"])
        options = json.loads(client.complete(prompt).text)
        if not isinstance(options, list):
            raise SystemExit(f"\nERROR: expected a JSON list of paraphrases, got: {options!r}\n")
        options = [str(p) for p in options]
        accepted = options if args.accept_all else review(q["query_id"], q["text"], options)
        for j, text in enumerate(accepted, 1):
            items.append({"query_id": f"{q['query_id']}_P{j:02d}", "source_query_id": q["query_id"],
                          "text": text, "source": f"paraphrase:{client.model}",
                          "knowledge_axis": q.get("knowledge_axis", "unassigned"), "layer": 2})
        write_json(out, items)
    print(f"\nLayer 2: {len(items)} paraphrases -> {out}")


if __name__ == "__main__":
    main()
