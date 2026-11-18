"""Step 9 — Score every RAG answer with RAGAS (Lens A: answer quality).

Primary metrics (every layer): context recall (did retrieval fetch what the
reference answer needs?) and answer correctness (does the answer match the
reference?). Layers listed in ``ragas.all_metrics_layers`` also get
faithfulness, context precision and answer relevancy. Layer 2 paraphrases are
scored against their parent query's reference.

The judge (``ragas.judge_model``, an OpenAI model) runs at temperature 0
(reasoning models only accept their default); embeddings use
``ragas.embeddings_model``. Pin the ragas version (the paper used 0.4.3) —
prompts change between versions. A failed metric is recorded as NaN.

Writes ``scores/ragas_scores.csv``, saved after every row; scored
(query, corpus) pairs are skipped on the next run.

    python evaluation/retrieval_rag/score_ragas.py --settings <file> [--layers 1 2 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import (  # noqa: E402
    corpora_names,
    load_queries,
    load_references,
    paraphrase_parents,
    rag_output_file,
    ragas_csv,
)
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402

ALL_METRICS = ["context_recall", "answer_correctness", "faithfulness", "context_precision", "answer_relevancy"]


def is_reasoning_model(model: str) -> bool:
    """o-series and gpt-5+ (ragas mis-detects decimal versions such as gpt-5.5)."""
    m = model.lower()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if m.startswith("gpt-"):
        try:
            return int(m[4:].split("-")[0].split("_")[0].split(".")[0]) >= 5
        except ValueError:
            return False
    return False


def ragas_judge(model: str, embeddings_model: str):
    """RAGAS LLM and embeddings through the ragas 0.4 llm_factory API."""
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("\nERROR: OPENAI_API_KEY is not set (needed for the RAGAS judge).\n")
    client = AsyncOpenAI(api_key=key)
    llm = llm_factory(model, client=client)
    if is_reasoning_model(model):
        llm.model_args.pop("max_tokens", None)
        llm.model_args.pop("top_p", None)
        llm.model_args["temperature"] = 1.0
        llm.model_args["max_completion_tokens"] = 4096
    else:
        llm.model_args["temperature"] = 0.0
        llm.model_args["max_tokens"] = 4096  # ragas' default 1024 truncates the claim lists
    return llm, OpenAIEmbeddings(model=embeddings_model, client=client)


def metric_objects(llm, emb, names):
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    build = {"context_recall": lambda: ContextRecall(llm=llm),
             "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=emb),
             "faithfulness": lambda: Faithfulness(llm=llm),
             "context_precision": lambda: ContextPrecision(llm=llm),
             "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=emb)}
    return {n: build[n]() for n in names}


INPUTS = {
    "context_recall": lambda r: dict(user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference"]),
    "answer_correctness": lambda r: dict(user_input=r["question"], response=r["answer"], reference=r["reference"]),
    "faithfulness": lambda r: dict(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]),
    "context_precision": lambda r: dict(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"]),
    "answer_relevancy": lambda r: dict(user_input=r["question"], response=r["answer"]),
}


def build_rows(s, layers) -> list:
    refs, parents = load_references(s), paraphrase_parents(s)
    rows = []
    for q in load_queries(s, layers=layers):
        ref = refs.get(parents.get(q["query_id"], q["query_id"]))
        if ref is None:
            print(f"  No reference for {q['query_id']} — skipped")
            continue
        for corpus in corpora_names(s):
            p = rag_output_file(s, q["layer"], q["query_id"], corpus)
            if not p.exists():
                print(f"  Missing RAG output {p} — skipped")
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"query_id": q["query_id"], "layer": q["layer"], "condition": corpus,
                         "question": data["query_text"], "answer": data.get("llm_answer", ""),
                         "contexts": [c["chunk_text"] for c in data.get("retrieved_chunks", [])],
                         "reference": ref})
    return rows


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    s = load_settings(args.settings)
    cfg = s.get("retrieval_rag.ragas")
    primary = list(cfg.get("primary_metrics", ["context_recall", "answer_correctness"]))
    full_layers = set(cfg.get("all_metrics_layers", [1]))

    out = ragas_csv(s)
    previous = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["query_id"], r["condition"]) for r in previous}
    rows = [r for r in build_rows(s, args.layers) if (r["query_id"], r["condition"]) not in done]
    print(f"{len(rows)} (query, corpus) pairs to score; {len(done)} already scored.")
    if args.dry_run or not rows:
        return

    import ragas

    llm, emb = ragas_judge(cfg["judge_model"], cfg["embeddings_model"])
    metrics = metric_objects(llm, emb, ALL_METRICS)
    results = list(previous)
    for r in rows:
        names = ALL_METRICS if r["layer"] in full_layers else primary
        rec = {"query_id": r["query_id"], "layer": r["layer"], "condition": r["condition"]}
        for name in ALL_METRICS:
            if name not in names:
                rec[name] = float("nan")
                continue
            try:
                rec[name] = float(metrics[name].score(**INPUTS[name](r)))
            except Exception as exc:  # noqa: BLE001 — record NaN, keep going
                print(f"  {r['query_id']} x {r['condition']}: {name} failed ({str(exc)[:100]})")
                rec[name] = float("nan")
        rec.update(judge_model=cfg["judge_model"], ragas_version=ragas.__version__)
        results.append(rec)
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"  scored {r['query_id']} x {r['condition']}")
    print(f"Scores: {out}\nNext: python evaluation/retrieval_rag/score_self_consistency.py")


if __name__ == "__main__":
    main()
