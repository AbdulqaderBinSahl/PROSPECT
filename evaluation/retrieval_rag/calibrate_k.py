"""Step 6 — Choose k: how many retrieved chunks the reference drafter sees.

For a few calibration queries (``references.calibration.queries``, which must
not appear in any layer), drafts a reference answer from the top-k chunks of
the unfiltered corpus for every k in ``references.calibration.k_values``, and
optionally has an auditor model score each draft for groundedness,
completeness and coherence (1-4).

Writes the drafts to ``retrieval_rag/calibration_k/`` and a table of draft
length and audit scores per k. Read the drafts, pick the k where more context
stops improving them, and set ``references.k``. (The paper chose k = 20.)

    python evaluation/retrieval_rag/calibrate_k.py --settings <file> [--no-audit]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import UNFILTERED, Retriever, format_reference_context, rr_path  # noqa: E402
from evaluation.shared.llm import client_for  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings, to_markdown, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--no-audit", action="store_true", help="Skip the auditor model")
    args = parser.parse_args()
    s = load_settings(args.settings)

    ref = s.get("retrieval_rag.references")
    cal = ref["calibration"]
    template = s.resolve(ref["prompt"]).read_text(encoding="utf-8")
    drafter = client_for(ref["drafter"])
    auditor = None if args.no_audit or not cal.get("auditor") else client_for(cal["auditor"])
    audit_template = s.resolve(cal.get("auditor_prompt", "evaluation/retrieval_rag/prompts/auditor.txt")) \
        .read_text(encoding="utf-8") if auditor else ""
    retriever = Retriever(s)

    rows = []
    for q in cal["queries"]:
        for k in cal["k_values"]:
            out = rr_path(s, "calibration_k", f"{q['query_id']}_k{k}.json")
            data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
            chunks = retriever.retrieve(UNFILTERED, q["text"], int(k))
            context = format_reference_context(chunks)
            if not data.get("draft"):
                prompt = template.format(query=q["text"], context=context, k=k,
                                         application_context=ref["application_context"].strip())
                data = {"query_id": q["query_id"], "query_text": q["text"], "k": k,
                        "drafter_model": drafter.model, "draft": drafter.complete(prompt).text,
                        "n_chunks": len(chunks)}
            if auditor and "audit" not in data:
                try:
                    data["audit"] = auditor.complete_json(
                        audit_template.format(query=q["text"], context=context, draft=data["draft"]))
                except Exception as exc:  # noqa: BLE001 — keep the draft, report the failure
                    print(f"  audit failed for {q['query_id']} k={k}: {exc}")
            write_json(out, data)
            audit = data.get("audit", {})
            rows.append({"query_id": q["query_id"], "k": k, "words": len(data["draft"].split()),
                         **{d: audit.get(d, "") for d in ("groundedness", "completeness", "coherence")},
                         "flagged_claims": len(audit.get("flagged_claims", []) or [])})
            print(f"  {q['query_id']} k={k}: {rows[-1]['words']} words")

    table = rr_path(s, "calibration_k", "summary.md")
    table.write_text("# k calibration\n\n" + to_markdown(rows) + "\n\nRead the drafts, choose k, "
                     "and set retrieval_rag.references.k.\n", encoding="utf-8")
    print(f"\nSummary: {table}")


if __name__ == "__main__":
    main()
