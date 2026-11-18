"""Step 8 — Answer every query from every corpus: retrieve top-k, then generate.

For each (query, corpus) pair: retrieve ``retrieval.top_k`` chunks from that
corpus's collection, format them as numbered evidence, and ask the generator
(``generator.model``) to answer from the evidence only, citing it. Saves
``rag_outputs/layer_<n>/<query_id>/<corpus>.json`` with the retrieved chunks
and the answer. Existing outputs are skipped (resumable).

    python evaluation/retrieval_rag/run_rag_grid.py --settings <file> [--layers 1 2 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import (  # noqa: E402
    Retriever,
    collection_name,
    corpora_names,
    format_generator_context,
    generator_prompt,
    load_queries,
    rag_output_file,
)
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    s = load_settings(args.settings)

    queries = load_queries(s, layers=args.layers)
    corpora = corpora_names(s)
    todo = [(q, c) for q in queries for c in corpora
            if not rag_output_file(s, q["layer"], q["query_id"], c).exists()]
    print(f"{len(queries)} queries x {len(corpora)} corpora; {len(todo)} pairs to run.")
    if args.dry_run or not todo:
        return

    k = int(s.get("retrieval_rag.retrieval.top_k"))
    retriever = Retriever(s)
    generator = client_for(s.get("retrieval_rag.generator.model"))
    errors = 0
    for i, (q, corpus) in enumerate(todo, 1):
        try:
            chunks = retriever.retrieve(corpus, q["text"], k)
            answer = generator.complete(generator_prompt(s, q["text"], format_generator_context(chunks))).text
        except Exception as exc:  # noqa: BLE001 — report; rerun to retry
            print(f"  [{i}/{len(todo)}] FAILED {q['query_id']} x {corpus}: {exc}")
            errors += 1
            continue
        write_json(rag_output_file(s, q["layer"], q["query_id"], corpus), {
            "query_id": q["query_id"], "layer": q["layer"], "query_text": q["text"],
            "condition": corpus, "collection": collection_name(s, corpus), "top_k": k,
            "generator_model": generator.model, "retrieved_chunks": chunks, "llm_answer": answer,
        })
        print(f"  [{i}/{len(todo)}] {q['query_id']} x {corpus}")
    print(f"Done; {errors} failed (run again to retry).\nNext: python evaluation/retrieval_rag/score_ragas.py")


if __name__ == "__main__":
    main()
