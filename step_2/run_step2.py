#!/usr/bin/env python3
"""
PROSPECT Step 2 — Application-conditioned corpus filtering.

Filters a folder of PDFs into keep / discard for the application described
in Step 1. Every condition in a run folder shares the same inputs and the
same operational config (criteria, rubrics, representative queries,
anti-patterns), which the configuration pre-step generates once.

Conditions (``--condition``):
    A3w  four specialist agents + application-conditioned chunk weights (default)
    A3   four specialist agents, uniform chunk weights
    A4   A3 without Agent 3 (Utility Simulator)
    A2   per chunk, all agent roles in one prompt, plus one aggregation call
    A1   whole paper, one call

Usage (from the repository root):

    # After Step 1: PDFs in runs/my_topic/papers/
    python step_2/run_step2.py --run-dir runs/my_topic
    python step_2/run_step2.py --run-dir runs/my_topic --condition A1

    # Without Step 1: your own intent brief and PDF folder
    python step_2/run_step2.py --run-dir runs/my_topic \\
        --input my_intent_brief.json --papers-dir path/to/pdfs

    # Generate and inspect the operational config before filtering
    python step_2/run_step2.py --run-dir runs/my_topic --prepare-only

    # Continue an interrupted run
    python step_2/run_step2.py --run-dir runs/my_topic --condition A3w --resume

Outputs go to ``<run-dir>/<condition>/``: per-paper decisions with reasoning
(``paper_decisions/``), the PDFs routed to ``routed_pdfs/keep`` and
``routed_pdfs/discard``, run metrics, and a run manifest.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import List

_STEP2_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STEP2_DIR.parent
for _p in (str(_REPO_ROOT), str(_STEP2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.env import load_env, resolve_llm_endpoint  # noqa: E402

load_env()

from baselines.a1_whole_paper import run_a1  # noqa: E402
from baselines.a2_all_roles import run_a2  # noqa: E402
from baselines.shared import BaselineLLM, write_manifest  # noqa: E402
from rag_corpus_filter.config import AgentModelMapping, PipelineConfig  # noqa: E402
from rag_corpus_filter.llm_utils import build_agent_clients, validate_models_available  # noqa: E402
from rag_corpus_filter.pipeline import run_pipeline  # noqa: E402
from rag_corpus_filter.run_setup import (  # noqa: E402
    CONDITIONS,
    DEFAULT_CONDITION,
    RunSetupError,
    canonical_condition,
    check_pdf_set,
    ensure_operational_config,
    log_operational_config,
    report_low_text_pdfs,
    report_pdf_matching,
    resolve_run_inputs,
)
from rag_corpus_filter.step1_loader import find_pdfs  # noqa: E402

logger = logging.getLogger("step2")

# Retry budgets for malformed LLM output, per condition family.
_A1_MAX_RETRIES = 8
_A2_MAX_RETRIES = 6


class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write() to avoid progress bar corruption."""

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm import tqdm
        msg = self.format(record)
        # Windows consoles may use cp1252 — replace un-encodable chars to avoid crashes
        encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
        tqdm.write(msg)


def _setup_logging(verbose: bool = False) -> None:
    """Configure tqdm-safe logging."""
    handler = _TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[handler])
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="PROSPECT Step 2 — application-conditioned corpus filtering")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run folder (shared with Step 1). Outputs go to <run-dir>/<condition>/")
    p.add_argument("--condition", default=DEFAULT_CONDITION,
                   help=f"One of {', '.join(CONDITIONS.values())} (default: {DEFAULT_CONDITION})")
    p.add_argument("--input", type=Path, default=None,
                   help="Step 1 corpus JSON or intent brief JSON. Default: Step 1 output found in "
                        "<run-dir>/step1/. Only needed on the first Step 2 call of a run folder.")
    p.add_argument("--papers-dir", type=Path, default=None,
                   help="Folder of PDFs to filter (default: <run-dir>/papers). "
                        "Only needed on the first Step 2 call of a run folder.")
    p.add_argument("--prepare-only", action="store_true",
                   help="Generate the run's operational config and stop (to review it before filtering)")
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run of this condition (skips papers already decided)")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    llm = p.add_argument_group("LLM endpoint (default: values from .env)")
    llm.add_argument("--base-url", help="OpenAI-compatible endpoint (overrides LLM_BASE_URL)")
    llm.add_argument("--api-key", help="API key (overrides LLM_API_KEY)")
    llm.add_argument("--model", help="Model ID for all agents (overrides LLM_MODEL)")
    llm.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    llm.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens per call (default: 4096)")

    agents = p.add_argument_group("Per-agent models (A3 / A3w / A4 only; default: --model)")
    agents.add_argument("--agent1-model", help="Agent 1 — Paper Profiler")
    agents.add_argument("--agent2-model", help="Agent 2 — Relevance Assessor")
    agents.add_argument("--agent3-model", help="Agent 3 — Utility Simulator")
    agents.add_argument("--agent4-model", help="Agent 4 — Aggregator Judge")
    agents.add_argument("--prestep-model", help="Configuration pre-step")
    return p.parse_args()


def main() -> None:
    """Run one Step 2 condition (or only the configuration pre-step)."""
    args = parse_args()
    _setup_logging(args.verbose)

    try:
        condition = canonical_condition(args.condition)
        inputs = resolve_run_inputs(args.run_dir, args.input, args.papers_dir)
    except RunSetupError as exc:
        sys.exit(f"\nERROR: {exc}\n")

    endpoint = resolve_llm_endpoint(args.base_url, args.api_key, args.model)

    config = PipelineConfig()
    config.llm.model = endpoint.model
    config.llm.base_url = endpoint.base_url
    config.llm.api_key = endpoint.api_key
    config.llm.temperature = args.temperature
    config.llm.max_tokens = args.max_tokens
    config.agent_models = AgentModelMapping(
        agent1_profiler=args.agent1_model,
        agent2_relevance=args.agent2_model,
        agent3_utility=args.agent3_model,
        agent4_aggregator=args.agent4_model,
        config_prestep=args.prestep_model,
    )
    out_dir = inputs.run_dir if args.prepare_only else inputs.condition_dir(condition)
    config.paths.output_dir = out_dir
    config.paths.papers_dir = inputs.papers_dir

    logger.info("Run folder : %s", inputs.run_dir)
    logger.info("Input      : %s (%s)", inputs.input_path, inputs.input_kind)
    logger.info("PDF folder : %s", inputs.papers_dir)
    logger.info("LLM        : %s @ %s", endpoint.model, endpoint.base_url)

    validate_models_available(config.llm, config.agent_models)
    agent_clients = build_agent_clients(config.llm, config.agent_models)

    # One operational config per run folder, shared by every condition.
    op_config = ensure_operational_config(inputs, config, agent_clients["config_prestep"])
    log_operational_config(op_config, inputs.operational_config_path)
    if args.prepare_only:
        print(f"\nOperational config ready: {inputs.operational_config_path}")
        print("Review it, then run a condition, e.g.:")
        print(f"  python step_2/run_step2.py --run-dir {args.run_dir} --condition {DEFAULT_CONDITION}\n")
        return

    pdfs = find_pdfs(inputs.papers_dir)
    if not pdfs:
        sys.exit(f"\nERROR: no PDFs found in {inputs.papers_dir}.\n")

    decisions_dir = out_dir / "paper_decisions"
    if decisions_dir.is_dir() and any(decisions_dir.glob("*.json")) and not args.resume:
        sys.exit(
            f"\nERROR: {out_dir} already has decisions for condition {condition}.\n"
            "Pass --resume to continue that run, or use a new --run-dir.\n"
        )

    check_pdf_set(inputs, pdfs)
    report_pdf_matching(inputs, pdfs)
    low_text = report_low_text_pdfs(pdfs)
    logger.info("Condition  : %s on %d PDFs -> %s", condition, len(pdfs), out_dir)

    if condition in ("A3", "A3w", "A4"):
        run_pipeline(
            config, agent_clients, inputs.step1_data, op_config, pdfs,
            condition=condition, input_info=inputs.manifest_info(), resume=args.resume,
            skip_utility=(condition == "A4"), adaptive_weights=(condition == "A3w"),
        )
    else:
        if any([args.agent1_model, args.agent2_model, args.agent3_model, args.agent4_model]):
            logger.warning("Per-agent model flags are ignored for %s (single-model baseline).", condition)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Same record as the A3 family: the config this condition used, beside its outputs.
        shutil.copyfile(inputs.operational_config_path, out_dir / "operational_config.json")
        llm = BaselineLLM(
            base_url=endpoint.base_url, api_key=endpoint.api_key, model=endpoint.model,
            temperature=args.temperature, max_tokens=args.max_tokens,
            max_retries=_A1_MAX_RETRIES if condition == "A1" else _A2_MAX_RETRIES,
        )
        write_manifest(out_dir, condition, llm, inputs.manifest_info(), len(pdfs), args.resume)
        if condition == "A1":
            run_a1(pdfs, inputs.operational_config_path, out_dir, llm, resume=args.resume)
        else:
            run_a2(pdfs, inputs.operational_config_path, out_dir, llm,
                   step1_data=inputs.step1_data, resume=args.resume)

    missing = papers_without_decision(pdfs, decisions_dir)
    print(f"\nDone. Kept papers: {out_dir / 'routed_pdfs' / 'keep'}")
    print(f"Per-paper decisions and reasoning: {decisions_dir}\n")
    if missing:
        print(f"WARNING: {len(missing)} of {len(pdfs)} papers have no decision because their LLM "
              "calls failed (see the log above):")
        for name in missing:
            print(f"  - {name}")
        print("Run the same command again with --resume to retry only these papers.\n")
    if low_text:
        print(f"NOTE: {len(low_text)} PDFs have almost no extractable text (probably scanned) "
              "and were judged on that text alone. Run OCR on them to have them assessed properly:")
        for name in low_text:
            print(f"  - {name}")
        print()


def papers_without_decision(pdfs: List[Path], decisions_dir: Path) -> List[str]:
    """Return the names of PDFs that have no decision file in ``decisions_dir``.

    A1/A2 name decisions ``<pdf stem>.json``; the A3 family appends a
    ``_YYYYMMDD_HHMMSS`` timestamp.
    """
    stems = {f.stem for f in decisions_dir.glob("*.json")} if decisions_dir.is_dir() else set()
    decided = stems | {re.sub(r"_\d{8}_\d{6}$", "", s) for s in stems}
    return [p.name for p in pdfs if p.stem not in decided]


if __name__ == "__main__":
    main()
