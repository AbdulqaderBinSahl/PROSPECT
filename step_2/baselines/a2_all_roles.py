"""Condition A2 -- All roles in one prompt, per-chunk filter.

Same per-chunk granularity as PROSPECT, but ALL four agent roles (profiler,
relevance assessor, utility simulator, judge reasoning) are collapsed into a
SINGLE LLM call per chunk.  Then ONE aggregation call synthesises all chunk
assessments into a final keep/discard decision.

This isolates the effect of *decomposing* roles into separate LLM calls
(PROSPECT A3 does this; A2 does not).  Comparing A2 vs A3 answers:
"Does calling each role separately improve filtering quality?"

To keep that comparison clean, each chunk prompt carries the same paper
context the multi-agent Paper Profiler (Agent 1) receives: the application
domain, the Step 1 metadata, the Scopus abstract (or the PDF's first pages
when the PDF has no Step 1 record), and the last pages.

Run it through ``step_2/run_step2.py --condition A2``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from tqdm import tqdm

from baselines.shared import (
    BaselineLLM,
    OperationalConfig,
    format_anti_patterns_for_prompt,
    format_criteria_for_prompt,
    format_queries_for_prompt,
    load_operational_config,
    route_pdf,
)
from rag_corpus_filter.paper_profiler import format_paper_context
from rag_corpus_filter.step1_loader import Step1Data, resolve_paper_context
from rag_corpus_filter.metrics_tracker import failed_papers
from rag_corpus_filter.text_extractor import (
    NO_TEXT_REASON,
    chunk_pages,
    extract_first_last_pages,
    extract_pages,
)

logger = logging.getLogger("filter_a2")


# -- Response schemas --------------------------------------------------------


class QueryAssessment(BaseModel):
    """Utility of a chunk for answering one representative query."""
    query: str
    utility_score_0_to_5: int = Field(..., ge=0, le=5)
    reasoning: str


class ChunkAllRolesAssessment(BaseModel):
    """Combined profiler + relevance + utility assessment for one chunk."""

    # Profiler role
    paper_type: Literal[
        "experimental", "review", "simulation",
        "case_study", "methods_development",
        "industry_report", "mixed"
    ]
    data_types_observed: List[str]
    domain_fit: Literal["strong", "partial", "weak"]
    scope_notes: str

    # Relevance role
    domain_relevance_0_to_5: int = Field(..., ge=0, le=5)
    criteria_scores: Dict[str, int]
    anti_pattern_flags: List[str]
    contains_quantitative_data: bool

    # Utility role
    query_assessments: List[QueryAssessment]
    overall_utility_0_to_5: int = Field(..., ge=0, le=5)
    best_served_queries: List[str]

    # Combined notes
    assessor_notes: str


class PaperDecision(BaseModel):
    """Aggregation decision across all chunk assessments."""
    decision: Literal["keep", "discard"]
    overall_score_1_to_10: int = Field(..., ge=1, le=10)
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]


# -- Chunk-level prompt (all roles) -----------------------------------------

CHUNK_SYSTEM_PROMPT = """\
You are an expert scientific literature assessor performing THREE roles \
simultaneously for a single chunk of a research paper:

**Role 1 — Paper Profiler:** Characterise the paper's type, data content, \
and domain fit based on the chunk text and the paper context (abstract / \
conclusion) provided.

**Role 2 — Relevance & Quality Assessor:** Score the chunk on domain \
relevance and each value criterion independently using the 0-5 rubrics.

**Role 3 — Utility Simulator:** For each representative query, score how \
useful this chunk would be if retrieved as RAG context to answer that query.

You will be given:
1. The application description.
2. Value criteria with 0-5 scoring rubrics.
3. Representative queries the corpus must serve.
4. Anti-patterns (red flags).
5. Paper context: abstract and conclusion text (for profiling).
6. One chunk of text from the paper.

Produce a SINGLE JSON object covering all three roles:
{
  "paper_type": "<experimental|review|simulation|case_study|methods_development|industry_report|mixed>",
  "data_types_observed": ["<data types visible in this chunk>"],
  "domain_fit": "<strong|partial|weak>",
  "scope_notes": "<1-2 sentences: what this chunk covers>",

  "domain_relevance_0_to_5": <0-5>,
  "criteria_scores": {"<criterion_name>": <0-5>, ...},
  "anti_pattern_flags": ["<any matching anti-patterns, or empty list>"],
  "contains_quantitative_data": <true|false>,

  "query_assessments": [
    {"query": "<query text>", "utility_score_0_to_5": <0-5>, "reasoning": "<1 sentence>"},
    ...
  ],
  "overall_utility_0_to_5": <0-5>,
  "best_served_queries": ["<queries scoring >= 3>"],

  "assessor_notes": "<1-2 sentence overall summary>"
}

RULES:
1. Score each value criterion INDEPENDENTLY using the anchors provided.
2. Produce exactly one query_assessment for EACH representative query.
3. domain_fit reflects the PAPER's fit (using context provided), not just the chunk.
4. paper_type reflects the PAPER's type (using context provided), not just the chunk.
5. Utility scores: 0=noise, 1=barely related, 2=tangential, 3=partially useful, \
4=substantially useful, 5=directly addresses the query.
6. If anti-patterns are detected, list them but only lower domain_relevance if \
the chunk contains ZERO relevant content.

Output ONLY the JSON object. No explanatory text before or after."""


def build_chunk_prompt(
    config: OperationalConfig,
    chunk_text_str: str,
    paper_context: str,
) -> str:
    criteria_text = format_criteria_for_prompt(config.value_criteria)
    queries_text = format_queries_for_prompt(config.representative_queries)
    anti_text = format_anti_patterns_for_prompt(config.anti_patterns)

    return f"""\
## Application Description
{config.application_description}

## Value Criteria (with 0-5 scoring rubrics)
{criteria_text}

## Representative Queries ({len(config.representative_queries)} total -- assess ALL)
{queries_text}

## Anti-Patterns (red flags)
{anti_text}

## Paper Context (domain, metadata, abstract, conclusion / last pages)
{paper_context}

## Chunk Text (assess this)
{chunk_text_str}

Assess this chunk in all three roles. Output ONLY the JSON object."""


# -- Aggregation prompt ------------------------------------------------------

AGGREGATION_SYSTEM_PROMPT = """\
You are a scientific corpus curation judge. You will receive chunk-level \
assessments for a paper where each chunk was evaluated for profiling, \
relevance, AND utility in a single pass.

Based on the chunk assessments, decide: **keep** or **discard**. There is no \
middle ground.

- **keep**: The paper contains substantial, directly useful content for the \
application across multiple dimensions (relevance + utility).
- **discard**: The paper lacks meaningful value, has limited coverage, or \
anti-patterns dominate.

Respond with ONLY a JSON object:
{
  "decision": "keep" | "discard",
  "overall_score_1_to_10": <integer 1-10>,
  "reasoning": "<2-3 sentence justification>",
  "strengths": ["<strength 1>", ...],
  "weaknesses": ["<weakness 1>", ...]
}"""


def build_aggregation_prompt(
    config: OperationalConfig,
    paper_id: str,
    chunk_assessments: List[dict],
) -> str:
    chunk_summaries = []
    for ca in chunk_assessments:
        chunk_summaries.append(
            f"  Chunk {ca['chunk_id']}: "
            f"type={ca['paper_type']}, domain_fit={ca['domain_fit']}, "
            f"relevance={ca['domain_relevance_0_to_5']}, "
            f"utility={ca['overall_utility_0_to_5']}, "
            f"quant_data={ca['contains_quantitative_data']}, "
            f"criteria={json.dumps(ca['criteria_scores'])}, "
            f"anti_patterns={ca['anti_pattern_flags']}, "
            f"best_queries={len(ca['best_served_queries'])}, "
            f"data_types={ca['data_types_observed']}, "
            f"scope: {ca['scope_notes']}, "
            f"notes: {ca['assessor_notes']}"
        )

    # Summary statistics
    n = len(chunk_assessments)
    avg_rel = sum(ca["domain_relevance_0_to_5"] for ca in chunk_assessments) / max(n, 1)
    avg_util = sum(ca["overall_utility_0_to_5"] for ca in chunk_assessments) / max(n, 1)
    quant_chunks = sum(1 for ca in chunk_assessments if ca["contains_quantitative_data"])

    all_queries_served = set()
    for ca in chunk_assessments:
        all_queries_served.update(ca["best_served_queries"])

    # Domain fit consensus
    fit_counts = {}
    for ca in chunk_assessments:
        fit_counts[ca["domain_fit"]] = fit_counts.get(ca["domain_fit"], 0) + 1
    fit_summary = ", ".join(f"{k}={v}" for k, v in sorted(fit_counts.items()))

    # Paper type consensus
    type_counts = {}
    for ca in chunk_assessments:
        type_counts[ca["paper_type"]] = type_counts.get(ca["paper_type"], 0) + 1
    type_summary = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))

    return f"""\
## Application Description
{config.application_description}

## Paper: {paper_id}
## Total chunks assessed: {n}

## Summary Statistics
- Average domain relevance: {avg_rel:.1f} / 5
- Average utility: {avg_util:.1f} / 5
- Chunks with quantitative data: {quant_chunks} / {n}
- Distinct queries served (score >= 3): {len(all_queries_served)} / {len(config.representative_queries)}
- Paper type consensus: {type_summary}
- Domain fit consensus: {fit_summary}

## Per-Chunk Assessments
{chr(10).join(chunk_summaries)}

Based on these assessments, decide: keep or discard. Output ONLY the JSON object."""


# -- Runner ------------------------------------------------------------------


def run_a2(
    pdfs: List[Path],
    operational_config_path: Path,
    output_dir: Path,
    llm: BaselineLLM,
    step1_data: Optional[Step1Data],
    resume: bool = False,
    chunk_words: int = 1400,
    overlap_words: int = 200,
    min_chunk_words: int = 300,
    profiler_pages: int = 2,
) -> Dict[str, Any]:
    """Filter ``pdfs`` with condition A2. Returns the final run metrics."""
    config = load_operational_config(str(operational_config_path))
    decisions_dir = output_dir / "paper_decisions"
    chunks_dir = output_dir / "chunk_assessments"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

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
    total_chunks = 0
    total_llm_calls = 0
    start_time = time.time()

    def _save_metrics(status: str = "in_progress") -> Dict[str, Any]:
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
            "condition": "A2",
            "status": status,
            "run_timestamp": datetime.now().isoformat(),
            "total_papers": len(pdfs),
            "total_papers_processed": n_ok,
            "total_papers_failed": len(failed_now),
            "failed_papers": failed_now,
            "keep": keep,
            "discard": discard,
            "papers_processed_this_session": len(results),
            "total_chunks": total_chunks + sum(r.get("n_chunks", 0) for r in prev_per_paper),
            "total_llm_calls": calls,
            "total_wall_time_sec": round(cumulative_elapsed, 1),
            "this_session_wall_time_sec": round(session_elapsed, 1),
            "avg_time_per_paper_sec": round(paper_time / n_ok, 2) if n_ok else 0,
            "avg_llm_calls_per_paper": round(calls / n_ok, 2) if n_ok else 0,
            "hardware_model_info": llm.info(),
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "per_paper": all_per_paper,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return metrics

    for pdf_path in tqdm(pdfs, desc="Filter A2", unit="paper"):
        paper_id = pdf_path.stem

        if paper_id in existing_decisions:
            continue

        paper_start = time.time()
        paper_llm_calls = 0
        paper_n_chunks = 0
        paper_failed = False
        dec_data = None

        try:
            # Extract text, page by page (same extraction as A3)
            pages = extract_pages(pdf_path)
            if not any(page.text.strip() for page in pages):
                logger.warning("%s: %s Discarding.", paper_id, NO_TEXT_REASON)
                dec_data = {
                    "paper_id": paper_id,
                    "pdf": pdf_path.name,
                    "decision": "discard",
                    "overall_score_1_to_10": 1,
                    "reasoning": NO_TEXT_REASON,
                    "strengths": [],
                    "weaknesses": ["No extractable text"],
                    "n_chunks": 0,
                }
                dec_path = decisions_dir / f"{paper_id}.json"
                with open(dec_path, "w", encoding="utf-8") as f:
                    json.dump(dec_data, f, indent=2, ensure_ascii=False)
                route_pdf(pdf_path, "discard", output_dir / "routed_pdfs")
                results.append(dec_data)
                per_paper_metrics.append({
                    "pdf": pdf_path.name,
                    "decision": "discard",
                    "n_chunks": 0,
                    "total_llm_calls": 0,
                    "total_wall_time_sec": round(time.time() - paper_start, 2),
                    "failed": False,
                })
                _save_metrics("in_progress")
                continue

            # Paper context: identical to what Agent 1 receives in A3/A3-w/A4
            first_pages, last_pages = extract_first_last_pages(pdf_path, n_pages=profiler_pages)
            abstract_text, domain, metadata = resolve_paper_context(
                pdf_path, step1_data, first_pages
            )
            paper_context = format_paper_context(abstract_text, last_pages, domain, metadata)

            # Chunk the text (same chunker and settings as A3)
            chunks = chunk_pages(pages, chunk_words, overlap_words, min_chunk_words)
            paper_n_chunks = len(chunks)
            total_chunks += paper_n_chunks

            # Assess each chunk (all roles in one call)
            chunk_assessments = []
            for chunk in chunks:
                hp = build_chunk_prompt(config, chunk.text, paper_context)
                result = llm.call(
                    CHUNK_SYSTEM_PROMPT, hp, ChunkAllRolesAssessment,
                    context_label=f"a2_{paper_id}_{chunk.chunk_id}",
                )
                ca = result.model_dump()
                ca["chunk_id"] = chunk.chunk_id
                ca["word_count"] = chunk.word_count
                chunk_assessments.append(ca)
                paper_llm_calls += 1
                total_llm_calls += 1

            # Save chunk assessments
            ca_path = chunks_dir / f"{paper_id}_chunks.json"
            with open(ca_path, "w", encoding="utf-8") as f:
                json.dump(chunk_assessments, f, indent=2, ensure_ascii=False)

            # Aggregation call
            agg_prompt = build_aggregation_prompt(config, paper_id, chunk_assessments)
            decision = llm.call(
                AGGREGATION_SYSTEM_PROMPT, agg_prompt, PaperDecision,
                context_label=f"a2_agg_{paper_id}",
            )
            paper_llm_calls += 1
            total_llm_calls += 1

            # Save decision
            dec_data = decision.model_dump()
            dec_data["paper_id"] = paper_id
            dec_data["pdf"] = pdf_path.name
            dec_data["n_chunks"] = paper_n_chunks

            dec_path = decisions_dir / f"{paper_id}.json"
            with open(dec_path, "w", encoding="utf-8") as f:
                json.dump(dec_data, f, indent=2, ensure_ascii=False)

            # Route PDF
            route_pdf(pdf_path, decision.decision, output_dir / "routed_pdfs")

            results.append(dec_data)
            logger.info(
                "%s -> %s (score %d, %d chunks, %d LLM calls, %.1fs)",
                paper_id, decision.decision, decision.overall_score_1_to_10,
                paper_n_chunks, paper_llm_calls, time.time() - paper_start,
            )

        except Exception:
            logger.exception("Failed on %s", paper_id)
            failed.append(paper_id)
            paper_failed = True

        per_paper_metrics.append({
            "pdf": pdf_path.name,
            "decision": dec_data["decision"] if dec_data and not paper_failed else None,
            "n_chunks": paper_n_chunks,
            "total_llm_calls": paper_llm_calls,
            "total_wall_time_sec": round(time.time() - paper_start, 2),
            "failed": paper_failed,
        })

        _save_metrics("in_progress")

    # Final metrics
    final = _save_metrics("completed")

    elapsed = time.time() - start_time
    logger.info("-" * 60)
    logger.info("Filter A2 complete")
    logger.info("  Processed  : %d papers in %.1fs (%.1fs/paper)", len(results), elapsed, elapsed / max(len(results), 1))
    logger.info("  Chunks     : %d total", total_chunks)
    logger.info("  LLM calls  : %d (%.1f/paper)", total_llm_calls, total_llm_calls / max(len(results), 1))
    logger.info("  Keep       : %d", final["keep"])
    logger.info("  Discard    : %d", final["discard"])
    logger.info("  Failed     : %d", len(failed))
    logger.info("  Output     : %s", output_dir)
    logger.info("-" * 60)
    return final
