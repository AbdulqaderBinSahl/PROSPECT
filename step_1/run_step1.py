#!/usr/bin/env python3
"""
PROSPECT Step 1 — Corpus Discovery.

Elicits the user's intent through an LLM conversation, turns it into Scopus
Boolean queries, retrieves paper metadata, and triages the abstracts. All
outputs go to ``<run-dir>/step1/``:

    intent_brief.json     the confirmed application description (always written)
    corpus.json           Scopus metadata for every retrieved paper
    corpus_triaged.json   papers that passed abstract triage
    triage_report.json    PASS/SKIP verdict and reason for every paper
    pdf_checklist.csv     the PDFs to download and the filenames Step 2 expects
    run_log.md            conversation, queries, counts (for reporting)

Usage (from the repository root):

    # Full Step 1: conversation -> Scopus -> abstract triage
    python step_1/run_step1.py --run-dir runs/my_topic

    # Keep every retrieved paper (no abstract triage)
    python step_1/run_step1.py --run-dir runs/my_topic --skip-triage

    # Conversation only: write the intent brief, bring your own PDFs
    python step_1/run_step1.py --run-dir runs/my_topic --skip-scopus

    # Triage an existing corpus.json later
    python step_1/run_step1.py --run-dir runs/my_topic --triage-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_STEP1_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STEP1_DIR.parent
for _p in (str(_REPO_ROOT), str(_STEP1_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.env import LLMEndpoint, load_env, resolve_llm_endpoint  # noqa: E402

load_env()  # must run before rag_corpus_builder.config reads the Scopus keys

from openai import OpenAI  # noqa: E402

from rag_corpus_builder import config  # noqa: E402
from rag_corpus_builder.intent_elicitor import run_elicitation  # noqa: E402
from rag_corpus_builder.output_handler import (  # noqa: E402
    save_intent_brief,
    save_output,
    save_run_log,
    write_pdf_checklist,
)
from rag_corpus_builder.query_validator import validate_queries  # noqa: E402
from rag_corpus_builder.scopus_retriever import retrieve_papers  # noqa: E402
from rag_corpus_builder.strategy_generator import generate_strategy  # noqa: E402
from rag_corpus_builder.triage import run_triage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step1")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="PROSPECT Step 1 — Corpus Discovery")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run folder shared by Step 1 and Step 2 (outputs go to <run-dir>/step1/)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--skip-scopus", action="store_true",
                      help="Only run the intent conversation and write intent_brief.json "
                           "(no Scopus key needed; bring your own PDFs)")
    mode.add_argument("--triage-only", action="store_true",
                      help="Run abstract triage on an existing <run-dir>/step1/corpus.json")
    p.add_argument("--skip-triage", action="store_true",
                   help="Keep every retrieved paper (skip abstract triage)")
    p.add_argument("--base-url", help="LLM endpoint (overrides LLM_BASE_URL)")
    p.add_argument("--api-key", help="LLM API key (overrides LLM_API_KEY)")
    p.add_argument("--model", help="LLM model ID (overrides LLM_MODEL)")
    return p.parse_args()


def _check_scopus_key() -> None:
    """Exit with a clear message if no Scopus API key is configured."""
    if not config.SCOPUS_API_KEY:
        sys.exit(
            "\nERROR: SCOPUS_API_KEY is not set.\n"
            "Get a key at https://dev.elsevier.com/ and add it to .env (see .env.example),\n"
            "or run with --skip-scopus to write only the intent brief.\n"
        )


def _banner(endpoint: LLMEndpoint, scopus: bool) -> None:
    """Print the run header."""
    print("\n" + "#" * 60)
    print("#    PROSPECT Step 1 — Corpus Discovery".ljust(59) + "#")
    print("#" * 60)
    print(f"\n  LLM endpoint : {endpoint.base_url}")
    print(f"  LLM model    : {endpoint.model}")
    if scopus:
        print("  Scopus API   : key loaded")
        print("  Scopus Inst  : " + ("institution token loaded" if config.SCOPUS_INST_TOKEN
                                     else "none (personal key only)"))
    else:
        print("  Scopus API   : skipped")


def _next_steps(run_dir: Path, checklist: Path | None) -> None:
    """Tell the user what to do before Step 2."""
    print("\n" + "#" * 60)
    print("  Step 1 complete. Next:")
    if checklist:
        print(f"  1. Download the PDFs listed in {checklist}")
        print(f"     and save them in {run_dir / 'papers'} under the 'expected_filename' names")
        print("     (any filename that contains the DOI works).")
    else:
        print(f"  1. Put the PDFs you want to filter in {run_dir / 'papers'}")
    print(f"  2. python step_2/run_step2.py --run-dir {run_dir}")
    print("#" * 60 + "\n")


def main() -> None:
    """Run Step 1 in the selected mode."""
    args = parse_args()
    run_dir: Path = args.run_dir.resolve()
    out_dir = run_dir / "step1"

    if args.triage_only:
        corpus_path = out_dir / "corpus.json"
        if not corpus_path.exists():
            sys.exit(f"\nERROR: {corpus_path} not found. Run Step 1 with Scopus retrieval first.\n")
        endpoint = resolve_llm_endpoint(args.base_url, args.api_key, args.model)
        client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
        triaged = run_triage(corpus_path, client, endpoint.model, out_dir)
        _next_steps(run_dir, write_pdf_checklist(triaged, out_dir))
        return

    if (out_dir / "intent_brief.json").exists():
        sys.exit(
            f"\nERROR: {out_dir} already contains Step 1 outputs. "
            "Use a new --run-dir, or delete that folder to start over.\n"
        )

    if not args.skip_scopus:
        _check_scopus_key()
    endpoint = resolve_llm_endpoint(args.base_url, args.api_key, args.model)
    client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)
    _banner(endpoint, scopus=not args.skip_scopus)

    # Phase 1 – Intent Elicitation
    intent_brief, transcript = run_elicitation(client, endpoint.model)
    save_intent_brief(intent_brief, out_dir)

    if args.skip_scopus:
        save_run_log(intent_brief, transcript, endpoint.model, endpoint.base_url, out_dir)
        _next_steps(run_dir, checklist=None)
        return

    # Phase 2 – Search Strategy Generation
    strategy = generate_strategy(client, endpoint.model, intent_brief)

    # Phase 3 – Query Validation
    strategy, validation_results = validate_queries(client, endpoint.model, strategy, intent_brief)

    # Phase 4 – Scopus Retrieval
    papers, query_results = retrieve_papers(strategy)

    # Phase 5 – Save Output + run log
    corpus_path = save_output(
        intent_brief, strategy, validation_results, papers, query_results, out_dir
    )
    save_run_log(
        intent_brief, transcript, endpoint.model, endpoint.base_url, out_dir,
        strategy=strategy, validation_results=validation_results,
        papers=papers, query_results=query_results, corpus_filepath=corpus_path,
    )

    # Phase 6 – Abstract triage
    final_corpus = Path(corpus_path)
    if not args.skip_triage and papers:
        final_corpus = run_triage(final_corpus, client, endpoint.model, out_dir)

    _next_steps(run_dir, write_pdf_checklist(final_corpus, out_dir))


if __name__ == "__main__":
    main()
