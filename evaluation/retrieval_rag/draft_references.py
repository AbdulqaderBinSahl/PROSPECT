"""Step 7 — Draft one reference answer per layer 1 and layer 3 query.

The drafter (``references.drafter``) answers each query from the top-k
chunks of the **unfiltered** corpus (k = ``references.k``), citing papers and
saying when the evidence is insufficient. Layer 2 paraphrases reuse their
parent query's reference.

Use a drafter from a different model family than the RAG generator and the
RAGAS judge, so no model grades its own writing (the paper: Gemini drafter,
Claude generator, GPT judge). Read the drafts before scoring.

Note for reporting: references drawn from the unfiltered corpus favour it on
context recall, so read the RAGAS results as non-inferiority / retention.

Writes ``references/<query_id>.json``; existing files are kept unless --overwrite.

    python evaluation/retrieval_rag/draft_references.py --settings <file> [--overwrite] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import (  # noqa: E402
    UNFILTERED,
    Retriever,
    format_reference_context,
    load_queries,
    references_dir,
)
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List what would be drafted")
    args = parser.parse_args()
    s = load_settings(args.settings)

    ref = s.get("retrieval_rag.references")
    k = int(ref["k"])
    queries = load_queries(s, layers=(1, 3))
    out_dir = references_dir(s)
    todo = [q for q in queries if args.overwrite or not (out_dir / f"{q['query_id']}.json").exists()]
    print(f"{len(queries)} queries (layers 1 and 3), k = {k}; {len(todo)} to draft.")
    if args.dry_run or not todo:
        return

    template = s.resolve(ref["prompt"]).read_text(encoding="utf-8")
    drafter = client_for(ref["drafter"])
    retriever = Retriever(s)
    for q in todo:
        chunks = retriever.retrieve(UNFILTERED, q["text"], k)
        prompt = template.format(query=q["text"], context=format_reference_context(chunks), k=k,
                                 application_context=ref["application_context"].strip())
        draft = drafter.complete(prompt).text.strip()
        write_json(out_dir / f"{q['query_id']}.json", {
            "query_id": q["query_id"], "layer": q["layer"], "query_text": q["text"],
            "drafter_model": drafter.model, "k": k, "draft": draft, "n_chunks": len(chunks),
            "chunk_ids": [c.get("paper_id", "") for c in chunks],
        })
        print(f"  {q['query_id']}: {len(draft.split())} words")
    print(f"References: {out_dir}\nNext: python evaluation/retrieval_rag/run_rag_grid.py")


if __name__ == "__main__":
    main()
