"""Step 11 — Gold labels (Lens B): is each paper relevant to each query?

A labeller model (``gold.labeller``; the paper: Gemini Flash) labels every
(layer 1 or 3 query, unfiltered paper) pair 0 / 1 / 2 from the paper's title
and first ``gold.excerpt_chars`` characters:

  2  directly answers — data, results, mechanisms or analysis an expert would cite
  1  tangentially related
  0  not relevant

The labeller never sees reference answers, RAG outputs or filter decisions,
so the gold sets are independent of Lens A and of the filters. Layer 2 is
excluded (a paraphrase has its parent's gold set).

Modes:
  --mode bulk        all pairs -> gold/gold_labels.jsonl
  --mode spot-check  re-label the spot-check pairs (build_gold_spot_check.py)
                     with ``gold.spot_check.second_model`` -> gold/spot_check/second_model_labels.jsonl

Resumable; ``--max-calls`` caps one session.

    python evaluation/retrieval_rag/label_gold.py --settings <file> [--mode bulk|spot-check] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import UNFILTERED, load_corpus, load_queries, rr_path  # noqa: E402
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, append_jsonl, load_settings, read_jsonl  # noqa: E402


def gold_file(s) -> Path:
    return rr_path(s, "gold", "gold_labels.jsonl")


def spot_key_file(s) -> Path:
    return rr_path(s, "gold", "spot_check", "KEY.csv")


def gold_prompt(template: str, field: str, query: str, paper: dict, n_chars: int) -> str:
    return (template.replace("{{ field }}", field).replace("{{ query }}", query)
            .replace("{{ title }}", paper["title"]).replace("{{ excerpt }}", paper["full_text"][:n_chars]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--mode", choices=["bulk", "spot-check"], default="bulk")
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    s = load_settings(args.settings)
    g = s.get("retrieval_rag.gold")

    queries = {q["query_id"]: q for q in load_queries(s, layers=(1, 3))}
    corpus = load_corpus(s, UNFILTERED)
    if args.mode == "bulk":
        pairs = [(qid, pid) for qid in queries for pid in corpus]
        out, spec = gold_file(s), g["labeller"]
    else:
        if not spot_key_file(s).exists():
            raise SystemExit("\nERROR: run build_gold_spot_check.py first.\n")
        with spot_key_file(s).open(encoding="utf-8") as fh:
            pairs = [(r["query_id"], r["paper_id"]) for r in csv.DictReader(fh)]
        out, spec = rr_path(s, "gold", "spot_check", "second_model_labels.jsonl"), g["spot_check"]["second_model"]

    done = {(r["query_id"], r["paper_id"]) for r in read_jsonl(out)}
    todo = [p for p in pairs if p not in done]
    print(f"{args.mode}: {len(pairs)} pairs, {len(done)} done, {len(todo)} to label with {spec['model']}.")
    if args.dry_run or not todo:
        return

    template = s.resolve(g["prompt"]).read_text(encoding="utf-8")
    n_chars, field = int(g.get("excerpt_chars", 6000)), g["field"]
    client = client_for(spec)
    made = 0
    for qid, pid in todo:
        if args.max_calls and made >= args.max_calls:
            print(f"Stopped at --max-calls={args.max_calls}; run again to continue.")
            break
        try:
            res = client.complete_json(gold_prompt(template, field, queries[qid]["text"], corpus[pid], n_chars))
            label = int(res.get("label"))
            if label not in (0, 1, 2):
                raise ValueError(f"bad label {res.get('label')!r}")
        except Exception as exc:  # noqa: BLE001 — skip; rerun to retry
            print(f"  FAILED {qid}/{pid}: {exc}")
            continue
        append_jsonl(out, {"query_id": qid, "layer": queries[qid]["layer"], "paper_id": pid, "label": label,
                           "rationale": res.get("rationale", ""), "judge_model": client.model})
        made += 1
        if made % 50 == 0:
            print(f"  {made} labelled")
    remaining = len(pairs) - len(read_jsonl(out))
    print(f"Labels: {out}" + (f" ({remaining} still to do; run again)" if remaining > 0 else ""))


if __name__ == "__main__":
    main()
