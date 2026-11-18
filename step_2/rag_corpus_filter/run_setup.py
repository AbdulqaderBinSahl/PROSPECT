"""
Run-folder setup shared by every Step 2 condition.

A run folder fixes the inputs once, so all conditions (A1, A2, A3, A3-w, A4)
filter the same papers with the same application context:

    <run-dir>/
        step1/                     Step 1 outputs (optional)
        papers/                    the PDFs to filter (default location)
        run_inputs.json            which input file and PDF folder this run uses
        operational_config.json    criteria, rubrics, queries, anti-patterns
        A3w/  A3/  A4/  A2/  A1/   one output folder per condition

``run_inputs.json`` is written by the first Step 2 call. Later calls reuse it
and refuse a different input file or PDF folder, and refuse to run if the
input file was edited afterwards. ``operational_config.json`` is generated
once by the configuration pre-step and read by every condition.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rag_corpus_filter.models import OperationalConfig
from rag_corpus_filter.text_extractor import (
    LOW_TEXT_WORDS_PER_PAGE,
    extract_pages,
    words_per_page,
)
from rag_corpus_filter.step1_loader import (
    Step1Data,
    detect_input_kind,
    load_run_input,
    match_pdf_to_paper,
)

logger = logging.getLogger(__name__)

# Canonical condition names, keyed by their upper-case form for lookup.
CONDITIONS = {
    "A1": "A1",    # whole paper, one call
    "A2": "A2",    # per chunk, all roles in one prompt
    "A3": "A3",    # per chunk, four specialist agents, uniform weights
    "A3W": "A3w",  # A3 + application-conditioned chunk weights (default)
    "A4": "A4",    # A3 without the Utility Simulator
}
DEFAULT_CONDITION = "A3w"

LOCK_NAME = "run_inputs.json"
OPERATIONAL_CONFIG_NAME = "operational_config.json"
STEP1_CANDIDATES = (
    "step1/corpus_triaged.json",
    "step1/corpus.json",
    "step1/intent_brief.json",
)


class RunSetupError(RuntimeError):
    """Raised when a run folder's inputs are missing or inconsistent."""


def canonical_condition(name: str) -> str:
    """Normalise a user-typed condition name (e.g. ``a3W`` -> ``A3w``)."""
    key = name.strip().upper().replace("-", "")
    if key not in CONDITIONS:
        raise RunSetupError(
            f"Unknown condition {name!r}. Choose one of: {', '.join(CONDITIONS.values())}"
        )
    return CONDITIONS[key]


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_lock_path(path: Path, run_dir: Path) -> str:
    """Store paths inside the run folder relative to it, so the folder can be moved."""
    path = path.resolve()
    try:
        return path.relative_to(run_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _from_lock_path(value: str, run_dir: Path) -> Path:
    """Inverse of :func:`_to_lock_path`."""
    p = Path(value)
    return p if p.is_absolute() else (run_dir / p).resolve()


@dataclass
class RunInputs:
    """The resolved, locked inputs of a run folder."""
    run_dir: Path
    input_path: Path
    input_kind: str          # "corpus" or "intent_brief"
    papers_dir: Path
    step1_data: Step1Data

    @property
    def lock_path(self) -> Path:
        return self.run_dir / LOCK_NAME

    @property
    def operational_config_path(self) -> Path:
        return self.run_dir / OPERATIONAL_CONFIG_NAME

    def condition_dir(self, condition: str) -> Path:
        return self.run_dir / condition

    def manifest_info(self) -> dict:
        """Input description recorded in each condition's run manifest."""
        return {
            "run_dir": str(self.run_dir),
            "input_path": str(self.input_path),
            "input_kind": self.input_kind,
            "papers_dir": str(self.papers_dir),
            "operational_config": str(self.operational_config_path),
            "domain": self.step1_data.domain,
            "n_paper_records": len(self.step1_data.papers),
        }


def _discover_input(run_dir: Path) -> Path:
    """Find Step 1 output inside the run folder (triaged corpus first)."""
    for rel in STEP1_CANDIDATES:
        candidate = run_dir / rel
        if candidate.exists():
            return candidate
    raise RunSetupError(
        f"No Step 1 output found in {run_dir / 'step1'}.\n"
        "Run Step 1 first, or pass --input with a Step 1 corpus JSON or an intent brief "
        "(see templates/intent_brief.example.json)."
    )


def resolve_run_inputs(
    run_dir: Path,
    input_path: Optional[Path] = None,
    papers_dir: Optional[Path] = None,
) -> RunInputs:
    """Create or reload the run folder's locked inputs.

    First call: picks the input (``input_path`` or Step 1 output found in the
    run folder) and the PDF folder (``papers_dir`` or ``<run-dir>/papers``),
    and writes ``run_inputs.json``. Later calls: reuse the locked values and
    raise :class:`RunSetupError` if different ones are passed or the input
    file has changed since it was locked.
    """
    run_dir = run_dir.resolve()
    lock_path = run_dir / LOCK_NAME

    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        locked_input = _from_lock_path(lock["input_path"], run_dir)
        locked_papers = _from_lock_path(lock["papers_dir"], run_dir)
        if input_path is not None and input_path.resolve() != locked_input:
            raise RunSetupError(
                f"This run folder uses {locked_input} as input (see {lock_path}).\n"
                "All conditions in a run folder must share the same input. "
                "Drop --input, or use a new --run-dir."
            )
        if papers_dir is not None and papers_dir.resolve() != locked_papers:
            raise RunSetupError(
                f"This run folder filters the PDFs in {locked_papers} (see {lock_path}).\n"
                "All conditions in a run folder must share the same PDFs. "
                "Drop --papers-dir, or use a new --run-dir."
            )
        if not locked_input.exists():
            raise RunSetupError(f"The locked input file {locked_input} no longer exists.")
        if _sha256(locked_input) != lock["input_sha256"]:
            raise RunSetupError(
                f"{locked_input} has changed since this run folder was created.\n"
                "Conditions compared within a run must use identical inputs. "
                "Restore the file or start a new --run-dir."
            )
        input_path, papers_dir = locked_input, locked_papers
        input_kind = lock["input_kind"]
    else:
        input_path = (input_path or _discover_input(run_dir)).resolve()
        if not input_path.exists():
            raise RunSetupError(f"Input file not found: {input_path}")
        input_kind = detect_input_kind(input_path)
        papers_dir = (papers_dir or run_dir / "papers").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = {
            "created_at": datetime.now().isoformat(),
            "input_path": _to_lock_path(input_path, run_dir),
            "input_kind": input_kind,
            "input_sha256": _sha256(input_path),
            "papers_dir": _to_lock_path(papers_dir, run_dir),
            "pdf_files": None,  # recorded by the first condition that runs
        }
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        logger.info("Created %s (input: %s)", lock_path, input_path)

    step1_data = load_run_input(input_path)
    return RunInputs(run_dir, input_path, input_kind, papers_dir, step1_data)


def check_pdf_set(inputs: RunInputs, pdfs: List[Path]) -> None:
    """Record the PDF set on the first condition run; warn if it changes later.

    Conditions are only comparable when they filtered the same PDFs.
    """
    lock = json.loads(inputs.lock_path.read_text(encoding="utf-8"))
    names = sorted(p.name for p in pdfs)
    if not lock.get("pdf_files"):
        lock["pdf_files"] = names
        inputs.lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        return
    recorded = set(lock["pdf_files"])
    added = sorted(set(names) - recorded)
    removed = sorted(recorded - set(names))
    if added or removed:
        logger.warning(
            "The PDF folder differs from the first condition run in this folder "
            "(%d added, %d removed). Conditions are only comparable on the same PDFs. "
            "Added: %s  Removed: %s",
            len(added), len(removed), added[:5], removed[:5],
        )


def report_pdf_matching(inputs: RunInputs, pdfs: List[Path]) -> None:
    """Log how many PDFs matched a Step 1 record (and so get the Scopus abstract)."""
    data = inputs.step1_data
    if not data.papers:
        logger.info(
            "No paper metadata in the input: every condition reads the abstract "
            "from each PDF's first pages."
        )
        return
    matched = [p for p in pdfs if match_pdf_to_paper(p, data) is not None]
    unmatched = [p.name for p in pdfs if p not in matched]
    logger.info(
        "PDF matching: %d / %d PDFs matched a Step 1 record by DOI or Scopus ID.",
        len(matched), len(pdfs),
    )
    if unmatched:
        logger.warning(
            "%d PDFs did not match any Step 1 record; for these, the first pages stand in "
            "for the abstract. Rename them as in step1/pdf_checklist.csv to fix this. "
            "First few: %s", len(unmatched), unmatched[:10],
        )


def report_low_text_pdfs(pdfs: List[Path]) -> List[str]:
    """Warn about PDFs with almost no extractable text and return their names.

    These are usually scanned images. PROSPECT does not run OCR, so such a
    paper is judged on the little text it has (or discarded when it has none).
    """
    low = [
        p.name for p in pdfs
        if words_per_page(extract_pages(p)) < LOW_TEXT_WORDS_PER_PAGE
    ]
    if low:
        logger.warning(
            "%d PDFs have almost no extractable text (under %d words per page), probably "
            "scanned images: %s. PROSPECT does not run OCR, so these will most likely be "
            "discarded. Run OCR on them first (for example with OCRmyPDF), or remove them.",
            len(low), LOW_TEXT_WORDS_PER_PAGE, low[:10],
        )
    return low


def load_operational_config(path: Path) -> OperationalConfig:
    """Load and validate an operational_config.json."""
    return OperationalConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def ensure_operational_config(inputs: RunInputs, config, prestep_llm) -> OperationalConfig:
    """Return the run's operational config, generating it once if missing.

    ``config`` is the PipelineConfig (for retries and the bad-output folder)
    and ``prestep_llm`` the client used by the configuration pre-step.
    """
    path = inputs.operational_config_path
    if path.exists():
        op = load_operational_config(path)
        logger.info("Using the run's operational config: %s", path)
        return op

    from rag_corpus_filter.config_prestep import run_config_prestep
    from rag_corpus_filter.output_handler import save_operational_config

    data = inputs.step1_data
    logger.info("No operational config yet — running the configuration pre-step...")
    op = run_config_prestep(
        prestep_llm, config, data.application_description,
        knowledge_axes=data.knowledge_axes,
        scope_boundaries=data.scope_boundaries,
        content_anti_patterns=data.content_anti_patterns,
        depth_expectation=data.depth_expectation,
    )
    save_operational_config(op, inputs.run_dir)
    logger.info("Saved %s — every condition in this run folder will use it.", path)
    return op


def log_operational_config(op: OperationalConfig, path: Path) -> None:
    """Print a short summary of the operational config."""
    logger.info("=" * 60)
    logger.info("OPERATIONAL CONFIG: %s", path)
    logger.info("  Queries: %d", len(op.representative_queries))
    logger.info("  Criteria: %s", [c.name for c in op.value_criteria])
    logger.info("  Taxonomy: %s", op.domain_taxonomy)
    logger.info("  Anti-patterns: %d", len(op.anti_patterns))
    logger.info(
        "  Evidence gate: min_criteria=%d, score_threshold=%d, min_strong=%d",
        op.strong_chunk_min_criteria,
        op.strong_chunk_score_threshold,
        op.min_strong_chunks_for_keep,
    )
    logger.info("=" * 60)
