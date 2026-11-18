"""
Pipeline orchestrator for the PROSPECT multi-agent conditions (A3, A3-w, A4).

Ties together all agents, the deterministic layer, and the output handler
into a single sequential execution flow. The run's inputs and operational
config are resolved by ``run_setup`` before this is called, so every
condition in a run folder sees identical inputs. Supports resume: if a
previous run was interrupted, re-running with resume=True skips papers that
already have a decision JSON.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from rag_corpus_filter.aggregator_judge import run_aggregator_judge
from rag_corpus_filter.config import PipelineConfig
from rag_corpus_filter.deterministic_layer import (
    apply_evidence_gate,
    build_deterministic_metrics,
)
from rag_corpus_filter.metrics_tracker import PaperMetrics, RunMetricsTracker
from rag_corpus_filter.models import (
    ChunkRelevanceQuality,
    ChunkUtilityAssessment,
    ChunkWeightsResponse,
    OperationalConfig,
    PaperDecision,
)
from rag_corpus_filter.output_handler import (
    build_chunk_row,
    build_paper_row,
    build_query_coverage_row,
    route_pdf,
    save_operational_config,
    save_paper_decision,
    write_chunk_jsonl,
    write_excel,
)
from rag_corpus_filter.paper_profiler import (
    run_chunk_weighting,
    run_paper_profiler,
    weights_as_dict,
)
from rag_corpus_filter.relevance_assessor import run_relevance_assessor
from rag_corpus_filter.step1_loader import (
    Step1Data,
    resolve_paper_context,
)
from rag_corpus_filter.text_extractor import (
    NO_TEXT_REASON,
    chunk_pages,
    extract_first_last_pages,
    extract_pages,
)
from rag_corpus_filter.utility_simulator import run_utility_simulator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _find_completed_papers(output_dir: Path) -> Set[str]:
    """Scan paper_decisions/ and return the set of PDF stems already processed.

    Decision files are named ``{pdf_stem}_{timestamp}.json``.  We strip the
    trailing ``_YYYYMMDD_HHMMSS`` to recover the original PDF stem.
    """
    decisions_dir = output_dir / "paper_decisions"
    if not decisions_dir.is_dir():
        return set()

    completed: Set[str] = set()
    for f in decisions_dir.glob("*.json"):
        # Filename pattern: {pdf_stem}_{YYYYMMDD}_{HHMMSS}.json
        # Strip the last two underscore-separated segments (date + time)
        parts = f.stem.rsplit("_", 2)
        if len(parts) >= 3:
            pdf_stem = "_".join(parts[:-2])
        else:
            pdf_stem = f.stem
        completed.add(pdf_stem)
    return completed


def _save_run_manifest(
    output_dir: Path,
    config: PipelineConfig,
    condition: str,
    input_info: Dict[str, Any],
    resume: bool,
    n_pdfs: int,
    skip_utility: bool = False,
    adaptive_weights: bool = False,
) -> Path:
    """Write a run_manifest.json capturing the invocation context.

    This fills the gap between operational_config.json (what the pipeline
    evaluates) and run_metrics.json (how it performed) — it records *how*
    the pipeline was invoked, so every run is fully reproducible.
    """
    manifest = {
        "run_started_at": datetime.now().isoformat(),
        "condition": condition,
        "skip_utility": skip_utility,
        "adaptive_weights": adaptive_weights,
        "inputs": input_info,
        "resume": resume,
        "papers_dir": str(config.paths.papers_dir),
        "output_dir": str(config.paths.output_dir),
        "n_pdfs_found": n_pdfs,
        "llm": {
            "default_model": config.llm.model,
            "base_url": config.llm.base_url,
            "temperature": config.llm.temperature,
            "max_retries": config.llm.max_retries,
        },
        "agent_model_mapping": {
            "agent1_profiler": config.agent_models.agent1_profiler or config.llm.model,
            "agent2_relevance": config.agent_models.agent2_relevance or config.llm.model,
            "agent3_utility": config.agent_models.agent3_utility or config.llm.model,
            "agent4_aggregator": config.agent_models.agent4_aggregator or config.llm.model,
            "config_prestep": config.agent_models.config_prestep or config.llm.model,
        },
        "chunking": {
            "chunk_words": config.chunking.chunk_words,
            "overlap_words": config.chunking.overlap_words,
            "min_chunk_words": config.chunking.min_chunk_words,
        },
        "profiler_n_pages": config.profiler_n_pages,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "argv": sys.argv,
        },
    }

    path = output_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Run manifest saved to %s", path)
    return path


def _discard_without_text(
    pdf_path: Path,
    config: PipelineConfig,
    paper_metrics: Optional[PaperMetrics],
) -> Dict[str, Any]:
    """Record a discard for a PDF with no extractable text, as A1 and A2 do.

    No LLM is called. Returns the row for the Papers sheet.
    """
    logger.warning("[%s] %s Discarding.", pdf_path.name, NO_TEXT_REASON)
    decision = PaperDecision(
        decision="discard", overall_score_1_to_10=1, main_reasoning=NO_TEXT_REASON,
        strengths=[], weaknesses=["No extractable text"],
        best_case_for_keeping="", best_case_for_discarding="", why_decision_wins="",
        agent_disagreements=[], recommended_tags=[], top_evidence=[],
    )
    if paper_metrics is not None:
        paper_metrics.paper_end = time.perf_counter()
        paper_metrics.decision = "discard"
    save_paper_decision(
        pdf_path.name, decision, config.paths.output_dir,
        paper_metrics=paper_metrics.to_dict() if paper_metrics is not None else None,
    )
    route_pdf(pdf_path, "discard", config.paths)
    return {
        "pdf": pdf_path.name, "decision": "discard", "overall_score_1_10": 1,
        "main_reasoning": NO_TEXT_REASON, "chunks_scanned": 0, "evidence_gate_note": "",
    }


def process_single_paper(
    pdf_path: Path,
    agent_clients: Dict[str, Any],
    config: PipelineConfig,
    operational_config: OperationalConfig,
    step1_data: Optional[Step1Data],
    jsonl_path: Path,
    paper_metrics: Optional[PaperMetrics] = None,
    skip_utility: bool = False,
    adaptive_weights: bool = False,
) -> tuple[
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
]:
    """Process a single paper through all pipeline stages.

    Parameters
    ----------
    agent_clients : dict with keys 'agent1', 'agent2', 'agent3', 'agent4'
        pointing to the ChatOpenAI instance for each agent role.

    Returns (paper_row, chunk_rows, coverage_row) or (None, [], None) on failure.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pdf_name = pdf_path.name
    label = pdf_path.stem

    # --- Text extraction ---
    pages = extract_pages(pdf_path)
    cs = config.chunking
    chunks = chunk_pages(pages, cs.chunk_words, cs.overlap_words, cs.min_chunk_words)
    if not chunks:  # no extractable text at all
        return _discard_without_text(pdf_path, config, paper_metrics), [], None

    first_text, last_text = extract_first_last_pages(
        pdf_path, n_pages=config.profiler_n_pages
    )

    if paper_metrics is not None:
        paper_metrics.n_chunks = len(chunks)

    logger.info("[%s] %d pages, %d chunks", pdf_name, len(pages), len(chunks))

    # --- Agent 1: Paper Profiler (on agent1 client) ---
    abstract_text, domain, metadata = resolve_paper_context(
        pdf_path, step1_data, first_text
    )
    t0 = time.perf_counter()
    profile = run_paper_profiler(
        agent_clients["agent1"], config, abstract_text, last_text, domain,
        metadata=metadata, context_label=f"{label}_profiler",
        paper_metrics=paper_metrics,
    )
    if paper_metrics is not None:
        paper_metrics.agent1_time = time.perf_counter() - t0
    logger.info(
        "[%s] Profile: type=%s, domain_fit=%s",
        pdf_name, profile.paper_type, profile.domain_fit_initial,
    )

    # --- Adaptive chunk weighting (A3-w condition) ---
    chunk_weights_response: Optional[ChunkWeightsResponse] = None
    chunk_weights: Optional[Dict[str, float]] = None
    if adaptive_weights:
        t_w = time.perf_counter()
        chunk_weights_response = run_chunk_weighting(
            agent_clients["agent1"], config, profile, chunks,
            operational_config,
            context_label=f"{label}_chunk_weights",
            paper_metrics=paper_metrics,
        )
        chunk_weights = weights_as_dict(chunk_weights_response)
        weight_time = time.perf_counter() - t_w
        if paper_metrics is not None:
            # Add weight time to Agent 1's total (it's the same LLM role)
            paper_metrics.agent1_time += weight_time

    # --- Agents 2 & 3: per-chunk assessment (PARALLEL) ---
    #
    # Agent 2 (relevance) and Agent 3 (utility) run on separate LLM
    # clients / models and can process each chunk concurrently.
    # When skip_utility=True (condition A4), Agent 3 is skipped entirely.
    agent2_results: List[ChunkRelevanceQuality] = [None] * len(chunks)  # type: ignore[list-item]
    agent3_results: List[ChunkUtilityAssessment] = [None] * len(chunks)  # type: ignore[list-item]

    a2_total_time = 0.0
    a3_total_time = 0.0

    def _run_a2(idx: int, chunk):
        return ("a2", idx, run_relevance_assessor(
            agent_clients["agent2"], config, chunk.text, operational_config,
            context_label=f"{label}_{chunk.chunk_id}_relevance",
            paper_metrics=paper_metrics,
        ))

    def _run_a3(idx: int, chunk):
        return ("a3", idx, run_utility_simulator(
            agent_clients["agent3"], config, chunk.text, operational_config,
            context_label=f"{label}_{chunk.chunk_id}_utility",
            paper_metrics=paper_metrics,
        ))

    if skip_utility:
        # A4 ablation: Agent 2 only, sequential
        for i, chunk in enumerate(chunks):
            logger.info(
                "[%s] Chunk %s (%d/%d) [A2 only, utility skipped]...",
                pdf_name, chunk.chunk_id, i + 1, len(chunks),
            )
            t_chunk = time.perf_counter()
            _, idx, result = _run_a2(i, chunk)
            agent2_results[idx] = result
            a2_total_time += time.perf_counter() - t_chunk
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i, chunk in enumerate(chunks):
                logger.info(
                    "[%s] Chunk %s (%d/%d)...",
                    pdf_name, chunk.chunk_id, i + 1, len(chunks),
                )
                t_chunk = time.perf_counter()

                fut_a2 = pool.submit(_run_a2, i, chunk)
                fut_a3 = pool.submit(_run_a3, i, chunk)

                for fut in as_completed([fut_a2, fut_a3]):
                    agent_tag, idx, result = fut.result()
                    if agent_tag == "a2":
                        agent2_results[idx] = result
                    else:
                        agent3_results[idx] = result

                elapsed_chunk = time.perf_counter() - t_chunk
                # Attribute roughly half each since they ran in parallel
                a2_total_time += elapsed_chunk / 2
                a3_total_time += elapsed_chunk / 2

    if paper_metrics is not None:
        paper_metrics.agent2_total_time = a2_total_time
        paper_metrics.agent3_total_time = a3_total_time

    # --- Deterministic aggregation ---
    a3_for_metrics = [] if skip_utility else agent3_results
    chunk_id_list = [c.chunk_id for c in chunks]
    metrics = build_deterministic_metrics(
        agent2_results, a3_for_metrics, operational_config,
        chunk_weights=chunk_weights, chunk_ids=chunk_id_list,
    )
    logger.info(
        "[%s] Metrics: composite=%.2f, strong_chunks=%d, coverage=%.1f%%, gap=%+.2f",
        pdf_name, metrics.composite_score_0_to_10, metrics.strong_chunk_count,
        metrics.query_coverage * 100, metrics.relevance_utility_gap,
    )

    # --- Agent 4: Aggregator Judge (on agent4 client) ---
    # A3-w: weights influence the Judge indirectly via the already-weighted
    # composite score and criterion averages in `metrics`. The per-chunk
    # summaries shown to the Judge are intentionally identical to A3 so the
    # ablation isolates the deterministic-weighting effect alone.
    t4 = time.perf_counter()
    a3_for_judge = [] if skip_utility else agent3_results
    decision = run_aggregator_judge(
        agent_clients["agent4"], config, profile, chunks,
        agent2_results, a3_for_judge,
        metrics, operational_config,
        context_label=f"{label}_judge",
        paper_metrics=paper_metrics,
    )
    if paper_metrics is not None:
        paper_metrics.agent4_time = time.perf_counter() - t4

    # --- Evidence gate ---
    final_decision, gate_note = apply_evidence_gate(
        decision, metrics.strong_chunk_count, operational_config
    )
    logger.info(
        "[%s] Decision: %s (Agent 4: %s, score=%d)%s",
        pdf_name, final_decision, decision.decision,
        decision.overall_score_1_to_10,
        f" [{gate_note}]" if gate_note else "",
    )

    # --- Save per-paper outputs ---
    if paper_metrics is not None:
        paper_metrics.paper_end = time.perf_counter()
        paper_metrics.decision = final_decision
    pm_dict = paper_metrics.to_dict() if paper_metrics is not None else None
    save_paper_decision(
        pdf_name, decision, config.paths.output_dir,
        profile=profile, metrics=metrics, paper_metrics=pm_dict,
        chunk_weights_detail=(
            chunk_weights_response.chunk_weights if chunk_weights_response else None
        ),
        application_description=(
            operational_config.application_description
            if chunk_weights_response else None
        ),
    )
    route_pdf(pdf_path, final_decision, config.paths)
    a3_for_output = [] if skip_utility else agent3_results
    write_chunk_jsonl(
        pdf_name, chunks, agent2_results, a3_for_output, jsonl_path,
        chunk_weights_detail=(
            chunk_weights_response.chunk_weights if chunk_weights_response else None
        ),
    )

    # --- Build rows for Excel ---
    paper_row = build_paper_row(
        pdf_name, decision, final_decision, gate_note,
        metrics, profile, len(chunks),
    )
    # Build a {chunk_id -> ChunkWeight} lookup for the Chunks sheet.
    weight_by_id = (
        {cw.chunk_id: cw for cw in chunk_weights_response.chunk_weights}
        if chunk_weights_response else {}
    )
    if skip_utility:
        chunk_rows = [
            build_chunk_row(pdf_name, c, a2, None, weight_by_id.get(c.chunk_id))
            for c, a2 in zip(chunks, agent2_results)
        ]
    else:
        chunk_rows = [
            build_chunk_row(pdf_name, c, a2, a3, weight_by_id.get(c.chunk_id))
            for c, a2, a3 in zip(chunks, agent2_results, agent3_results)
        ]
    coverage_row = build_query_coverage_row(
        pdf_name, metrics.query_coverage_detail, metrics.query_coverage,
    )

    return paper_row, chunk_rows, coverage_row


def run_pipeline(
    config: PipelineConfig,
    agent_clients: Dict[str, Any],
    step1_data: Step1Data,
    operational_config: OperationalConfig,
    pdfs: List[Path],
    condition: str,
    input_info: Dict[str, Any],
    resume: bool = False,
    skip_utility: bool = False,
    adaptive_weights: bool = False,
) -> Path:
    """Run a multi-agent PROSPECT condition (A3, A3-w or A4) over ``pdfs``.

    The inputs (``step1_data``, ``operational_config``, ``pdfs``) are the
    run folder's locked inputs, shared with every other condition. Set
    resume=True to skip papers that already have a decision JSON.

    Parameters
    ----------
    agent_clients : dict
        LLM client per role ('agent1'..'agent4'), from ``build_agent_clients``.
    condition : str
        Condition label recorded in the run manifest.
    input_info : dict
        Description of the run's inputs, recorded in the run manifest.
    skip_utility : bool
        If True, skip Agent 3 (Utility Simulator) — condition A4.
    adaptive_weights : bool
        If True, run adaptive chunk weighting via Profiler — condition A3-w.

    Returns the path to the generated Excel workbook.
    """
    config.paths.ensure_dirs()

    # Reuse existing JSONL when resuming, otherwise create a new one
    jsonl_path: Optional[Path] = None
    if resume:
        existing = sorted(config.paths.output_dir.glob("chunks_*.jsonl"))
        if existing:
            jsonl_path = existing[-1]  # most recent
            logger.info("Resuming: appending to existing %s", jsonl_path.name)
    if jsonl_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = config.paths.output_dir / f"chunks_{ts}.jsonl"

    # Keep a copy of the shared operational config next to this condition's
    # outputs, so each condition folder is self-describing.
    save_operational_config(operational_config, config.paths.output_dir)

    if adaptive_weights:
        logger.info("CONDITION A3-w (ACCS): Application-Conditioned Chunk Scoring ENABLED")
    if skip_utility:
        logger.info("CONDITION A4: Utility Simulator (Agent 3) SKIPPED")
    logger.info("Papers to process: %d PDFs found", len(pdfs))

    # --- Save run manifest (captures invocation context) ---
    # On resume, keep the original manifest and append a session entry
    # instead of overwriting, so the first invocation is always preserved.
    manifest_path = config.paths.output_dir / "run_manifest.json"
    if resume and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            sessions = existing.get("resume_sessions", [])
            sessions.append({
                "resumed_at": datetime.now().isoformat(),
                "argv": sys.argv,
                "n_pdfs_found": len(pdfs),
            })
            existing["resume_sessions"] = sessions
            manifest_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Appended resume session to existing run_manifest.json")
        except Exception:
            logger.warning("Could not update existing manifest, writing fresh one")
            _save_run_manifest(
                config.paths.output_dir, config,
                condition=condition, input_info=input_info,
                resume=resume, n_pdfs=len(pdfs),
                skip_utility=skip_utility,
                adaptive_weights=adaptive_weights,
            )
    else:
        _save_run_manifest(
            config.paths.output_dir, config,
            condition=condition, input_info=input_info,
            resume=resume, n_pdfs=len(pdfs),
            skip_utility=skip_utility,
            adaptive_weights=adaptive_weights,
        )

    # --- Resume: find already-completed papers ---
    completed_stems: Set[str] = set()
    if resume:
        completed_stems = _find_completed_papers(config.paths.output_dir)
        if completed_stems:
            logger.info(
                "Resuming: %d papers already processed, %d remaining",
                len(completed_stems),
                len(pdfs) - len(completed_stems),
            )

    # --- Process each paper ---
    tracker = RunMetricsTracker(condition=condition, n_pdfs=len(pdfs))
    all_paper_rows: List[Dict[str, Any]] = []
    all_chunk_rows: List[Dict[str, Any]] = []
    all_coverage_rows: List[Dict[str, Any]] = []
    all_metrics_rows: List[Dict[str, Any]] = []
    skipped = 0

    # On resume, load previous per-paper metrics so the final summary
    # and Excel Metrics sheet cover all papers across sessions.
    if resume:
        historical = tracker.load_previous(config.paths.output_dir)
        if historical:
            all_metrics_rows.extend(historical)
            logger.info("Loaded %d historical per-paper metrics from previous session(s)", len(historical))

    for pdf_path in tqdm(pdfs, desc="Papers"):
        if pdf_path.stem in completed_stems:
            skipped += 1
            continue

        pm = tracker.new_paper(pdf_path.name)
        try:
            paper_row, chunk_rows, coverage_row = process_single_paper(
                pdf_path, agent_clients, config, operational_config, step1_data,
                jsonl_path, paper_metrics=pm,
                skip_utility=skip_utility, adaptive_weights=adaptive_weights,
            )
            pm.paper_end = time.perf_counter()
            if paper_row:
                all_paper_rows.append(paper_row)
            all_chunk_rows.extend(chunk_rows)
            if coverage_row:
                all_coverage_rows.append(coverage_row)
            all_metrics_rows.append(pm.to_dict())
        except Exception:
            pm.paper_end = time.perf_counter()
            pm.failed = True
            logger.exception("[%s] Failed, skipping to next paper", pdf_path.name)
            all_metrics_rows.append(pm.to_dict())
            tracker.save_summary(config)
            continue

        # Log per-paper timing
        retries_str = f", retries={pm.total_retries}" if pm.total_retries else ""
        logger.info(
            "[%s] Done in %.1fs (A1: %.1fs, A2: %.1fs, A3: %.1fs, A4: %.1fs) "
            "| %d chunks, %d LLM calls%s",
            pdf_path.name, pm.total_wall_time,
            pm.agent1_time, pm.agent2_total_time,
            pm.agent3_total_time, pm.agent4_time,
            pm.n_chunks, pm.total_llm_calls, retries_str,
        )

        # Save run metrics incrementally (survives interrupts)
        tracker.save_summary(config)

    # --- Generate final Excel ---
    excel_path = write_excel(
        all_paper_rows, all_chunk_rows, all_coverage_rows,
        operational_config.representative_queries,
        config.paths.output_dir,
        metrics_rows=all_metrics_rows,
    )

    # --- Save run metrics summary ---
    metrics_path = tracker.save_summary(config)
    logger.info("Saved run metrics -> %s", metrics_path)

    # --- Summary ---
    decisions = [r.get("decision", "?") for r in all_paper_rows]
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Papers processed this run: %d", len(all_paper_rows))
    logger.info("  Papers skipped (already done): %d", skipped)
    logger.info("  Papers failed: %d", len(pdfs) - skipped - len(all_paper_rows))
    logger.info("  Keep: %d", decisions.count("keep"))
    logger.info("  Discard: %d", decisions.count("discard"))
    logger.info("  Total LLM calls: %d", sum(pm.total_llm_calls for pm in tracker.papers))
    logger.info("  Total retries: %d", sum(pm.total_retries for pm in tracker.papers))
    logger.info("  Excel: %s", excel_path)
    logger.info("  JSONL: %s", jsonl_path)
    logger.info("  Run metrics: %s", metrics_path)
    logger.info("=" * 60)

    return excel_path
