"""Condition A1 — Single-pass whole-paper filter.

One LLM call per paper. The LLM sees the full paper text plus the complete
operational config (value criteria with rubrics, representative queries,
anti-patterns). It produces a single keep/discard decision.

This is the simplest baseline: no chunking, no agent decomposition.
Run it through ``step_2/run_step2.py --condition A1``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

from pydantic import BaseModel
from tqdm import tqdm

from rag_corpus_filter.metrics_tracker import failed_papers
from rag_corpus_filter.text_extractor import NO_TEXT_REASON
from baselines.shared import (
    BaselineLLM,
    OperationalConfig,
    extract_text_from_pdf,
    format_anti_patterns_for_prompt,
    format_criteria_for_prompt,
    format_queries_for_prompt,
    load_operational_config,
    route_pdf,
)

logger = logging.getLogger("filter_a1")


# ── Response schema ──────────────────────────────────────────────────────────


class WholePaperDecision(BaseModel):
    decision: Literal["keep", "discard"]
    overall_score_1_to_10: int
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]


# ── Prompt construction ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert scientific literature assessor. Your task is to decide whether \
a research paper should be included in a specialised RAG corpus.

You will be given:
1. The application description — what the RAG system is for.
2. Value criteria with 0-5 scoring rubrics — what makes a paper valuable.
3. Representative queries the corpus must serve.
4. Anti-patterns — red flags that indicate a paper should be discarded.
5. The full text of the paper.

Based on ALL of the above, decide: **keep** or **discard**. There is no middle ground.

- **keep**: The paper contains substantial, directly useful content for the application.
- **discard**: The paper is not useful for the application, matches an anti-pattern, or has only tangential relevance.

Respond with ONLY a JSON object matching this exact schema:
{
  "decision": "keep" | "discard",
  "overall_score_1_to_10": <integer 1-10>,
  "reasoning": "<2-3 sentence justification>",
  "strengths": ["<strength 1>", ...],
  "weaknesses": ["<weakness 1>", ...]
}"""


def build_human_prompt(config: OperationalConfig, paper_text: str) -> str:
    criteria_text = format_criteria_for_prompt(config.value_criteria)
    queries_text = format_queries_for_prompt(config.representative_queries)
    anti_text = format_anti_patterns_for_prompt(config.anti_patterns)

    return f"""\
## Application Description
{config.application_description}

## Value Criteria (with 0-5 scoring rubrics)
{criteria_text}

## Representative Queries This Corpus Must Serve
{queries_text}

## Anti-Patterns (red flags for discard)
{anti_text}

## Paper Full Text
{paper_text}

Now assess this paper. Output ONLY the JSON object."""


# ── Runner ───────────────────────────────────────────────────────────────────


def run_a1(
    pdfs: List[Path],
    operational_config_path: Path,
    output_dir: Path,
    llm: BaselineLLM,
    resume: bool = False,
) -> Dict[str, Any]:
    """Filter ``pdfs`` with condition A1. Returns the final run metrics."""
    config = load_operational_config(str(operational_config_path))
    decisions_dir = output_dir / "paper_decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Found %d PDFs to process", len(pdfs))

    # Resume support — load previous metrics to accumulate across sessions
    existing_decisions = set()
    prev_per_paper: list = []
    prev_wall_time: float = 0.0
    metrics_path = output_dir / "run_metrics.json"
    if resume:
        for dec_file in decisions_dir.glob("*.json"):
            existing_decisions.add(dec_file.stem)
        if metrics_path.exists():
            try:
                prev_data = json.loads(metrics_path.read_text(encoding="utf-8"))
                prev_per_paper = prev_data.get("per_paper", [])
                prev_wall_time = prev_data.get("total_wall_time_sec", 0.0)
            except Exception:
                pass
        logger.info("Resume mode: %d papers already processed", len(existing_decisions))

    # Process
    results = []
    per_paper_metrics: list = []
    failed = []
    total_llm_calls = 0
    start_time = time.time()

    def _save_metrics(status: str = "in_progress") -> Dict[str, Any]:
        """Write run_metrics.json after every paper so it survives interruptions."""
        session_elapsed = time.time() - start_time
        cumulative_elapsed = prev_wall_time + session_elapsed
        all_per_paper = prev_per_paper + per_paper_metrics
        all_successful = [r for r in all_per_paper if not r.get("failed", False)]
        failed_now = failed_papers(all_per_paper)
        n_ok = len(all_successful)
        calls = sum(r.get("total_llm_calls", 0) for r in all_successful)
        paper_time = sum(r.get("total_wall_time_sec", 0) for r in all_successful)
        keep = sum(1 for r in all_successful if r.get("decision") == "keep")
        discard = sum(1 for r in all_successful if r.get("decision") == "discard")
        metrics = {
            "condition": "A1",
            "status": status,
            "run_timestamp": datetime.now().isoformat(),
            "total_papers": len(pdfs),
            "total_papers_processed": n_ok,
            "total_papers_failed": len(failed_now),
            "failed_papers": failed_now,
            "keep": keep,
            "discard": discard,
            "papers_processed_this_session": len(results),
            "total_llm_calls": calls,
            "total_wall_time_sec": round(cumulative_elapsed, 1),
            "this_session_wall_time_sec": round(session_elapsed, 1),
            "avg_time_per_paper_sec": round(paper_time / n_ok, 2) if n_ok else 0,
            "avg_llm_calls_per_paper": round(calls / n_ok, 2) if n_ok else 0,
            "hardware_model_info": llm.info(),
            "max_paper_words": "unlimited",
            "per_paper": all_per_paper,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    for pdf_path in tqdm(pdfs, desc="Filter A1", unit="paper"):
        paper_id = pdf_path.stem

        if paper_id in existing_decisions:
            logger.debug("Skipping %s (already processed)", paper_id)
            continue

        paper_start = time.time()
        paper_llm_calls = 0
        paper_failed = False
        dec_data = None

        try:
            # Extract text
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                logger.warning("%s: %s Discarding.", paper_id, NO_TEXT_REASON)
                decision = WholePaperDecision(
                    decision="discard",
                    overall_score_1_to_10=1,
                    reasoning=NO_TEXT_REASON,
                    strengths=[],
                    weaknesses=["No extractable text"],
                )
            else:
                # Build prompt and call LLM
                human_prompt = build_human_prompt(config, full_text)
                decision = llm.call(
                    SYSTEM_PROMPT, human_prompt, WholePaperDecision,
                    context_label=f"a1_{paper_id}",
                )
                paper_llm_calls = 1
                total_llm_calls += 1

            # Save decision
            dec_data = decision.model_dump()
            dec_data["paper_id"] = paper_id
            dec_data["pdf"] = pdf_path.name

            dec_path = decisions_dir / f"{paper_id}.json"
            with open(dec_path, "w", encoding="utf-8") as f:
                json.dump(dec_data, f, indent=2, ensure_ascii=False)

            # Route PDF
            route_pdf(pdf_path, decision.decision, output_dir / "routed_pdfs")

            results.append(dec_data)
            logger.info(
                "%s -> %s (score %d, %.1fs)",
                paper_id, decision.decision, decision.overall_score_1_to_10,
                time.time() - paper_start,
            )

        except Exception:
            logger.exception("Failed on %s", paper_id)
            failed.append(paper_id)
            paper_failed = True

        per_paper_metrics.append({
            "pdf": pdf_path.name,
            "decision": dec_data["decision"] if dec_data and not paper_failed else None,
            "total_wall_time_sec": round(time.time() - paper_start, 2),
            "total_llm_calls": paper_llm_calls,
            "failed": paper_failed,
        })

        # Save metrics after every paper (survives interruptions)
        _save_metrics("in_progress")

    # Final metrics
    final = _save_metrics("completed")

    elapsed = time.time() - start_time
    logger.info("─" * 60)
    logger.info("Filter A1 complete")
    logger.info("  Processed : %d papers in %.1fs (%.1fs/paper)", len(results), elapsed, elapsed / max(len(results), 1))
    logger.info("  LLM calls : %d", total_llm_calls)
    logger.info("  Keep      : %d", final["keep"])
    logger.info("  Discard   : %d", final["discard"])
    logger.info("  Failed    : %d", len(failed))
    logger.info("  Output    : %s", output_dir)
    logger.info("─" * 60)
    return final
