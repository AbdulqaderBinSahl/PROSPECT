"""Check an evaluation settings file before spending anything.

Reports, without calling any model: whether the run folder and every
condition's decisions are found, how many PDFs there are, which API keys the
configured models need and whether each is set (never its value), and, with
--services, whether Chroma and Ollama answer.

    python evaluation/check_setup.py --settings <file> [--services]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.shared.decisions import load_decisions  # noqa: E402
from evaluation.shared.llm import ModelSpec, missing_key  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402


def model_specs(s) -> dict:
    """Every model the settings use, by role."""
    out = {f"judge {j['id']}": j for j in s.get("filter_validity.llm_judges", []) or []}
    rr = s.get("retrieval_rag", {}) or {}
    pick = {
        "RAG generator": ("generator", "model"),
        "paraphraser": ("queries", "paraphrase", "model"),
        "reference drafter": ("references", "drafter"),
        "k-calibration auditor": ("references", "calibration", "auditor"),
        "independent judge": ("self_consistency", "judge"),
        "gold labeller": ("gold", "labeller"),
        "gold second model": ("gold", "spot_check", "second_model"),
    }
    for role, path in pick.items():
        node = rr
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if node:
            out[role] = node
    if rr.get("ragas"):
        out["RAGAS judge"] = {"provider": "openai", "model": rr["ragas"]["judge_model"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    parser.add_argument("--services", action="store_true", help="Also check Chroma and Ollama")
    args = parser.parse_args()
    s = load_settings(args.settings)
    problems = 0

    print(f"Settings : {s.source}\nWorkspace: {s.workspace}")
    try:
        print(f"Run      : {s.run_dir}")
        print(f"PDFs     : {len(list(s.papers_dir.glob('*.pdf')))} in {s.papers_dir}")
        for c in s.conditions:
            d = load_decisions(s.condition_dir(c))
            print(f"  {c:<6} {len(d):>4} decisions, {sum(v == 'keep' for v in d.values()):>4} kept")
    except (SystemExit, FileNotFoundError) as exc:
        print(f"  PROBLEM: {str(exc).strip()}")
        problems += 1

    print("\nModels and keys:")
    for role, spec in model_specs(s).items():
        try:
            m = ModelSpec.from_settings(spec)
        except ValueError as exc:
            print(f"  {role:<22} PROBLEM: {exc}")
            problems += 1
            continue
        absent = missing_key(m)
        print(f"  {role:<22} {m.label:<40} {'key missing: ' + absent if absent else 'ok'}")
        problems += bool(absent)

    if args.services:
        try:
            from evaluation.retrieval_rag.retrieval import store

            c = s.get("retrieval_rag.chroma")
            store.get_client(c.get("host", "localhost"), int(c.get("port", 8000))).heartbeat()
            print("\nChroma   : reachable")
        except Exception as exc:  # noqa: BLE001
            print(f"\nChroma   : PROBLEM ({exc})")
            problems += 1
        try:
            import ollama

            ollama.show(s.get("retrieval_rag.embedding.model"))
            print("Ollama   : embedding model available")
        except Exception as exc:  # noqa: BLE001
            print(f"Ollama   : PROBLEM ({exc})")
            problems += 1
    print(f"\n{problems} problem(s) found." if problems else "\nAll checks passed.")


if __name__ == "__main__":
    main()
