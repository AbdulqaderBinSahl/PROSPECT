"""Load a corpus (JSON from build_corpora.py, or a folder of text files) for indexing."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    paper_id: str
    title: str
    full_text: str


def load_papers(source: str) -> Dict[str, Paper]:
    """Load papers from a JSON file or a directory of text/markdown files.

    JSON format:
        {
            "paper_id": {"title": "...", "full_text": "..."},
            ...
        }

    Directory format:
        Each .txt or .md file is one paper.
        paper_id = filename stem.
        Title extracted from first line if it looks like a heading.
    """
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix == ".json":
        return _load_from_json(source_path)
    elif source_path.is_dir():
        return _load_from_directory(source_path)
    else:
        raise ValueError(
            f"Source must be a .json file or a directory of text files: {source}"
        )


# ── private helpers ──────────────────────────────────────────────────────────


def _load_from_json(path: Path) -> Dict[str, Paper]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers: Dict[str, Paper] = {}
    for paper_id, info in data.items():
        full_text = info.get("full_text", "")
        if not full_text.strip():
            logger.warning("Paper '%s' has empty full_text — skipping", paper_id)
            continue
        papers[paper_id] = Paper(
            paper_id=paper_id,
            title=info.get("title", paper_id),
            full_text=full_text,
        )

    logger.info("Loaded %d papers from JSON: %s", len(papers), path)
    return papers


def _load_from_directory(dir_path: Path) -> Dict[str, Paper]:
    valid_extensions = {".txt", ".md", ".text", ".markdown"}
    papers: Dict[str, Paper] = {}

    for file_path in sorted(dir_path.iterdir()):
        if file_path.suffix.lower() not in valid_extensions:
            continue

        paper_id = file_path.stem
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning("File '%s' is empty — skipping", file_path.name)
            continue

        title = _extract_title(text, fallback=paper_id)
        papers[paper_id] = Paper(
            paper_id=paper_id,
            title=title,
            full_text=text,
        )

    logger.info("Loaded %d papers from directory: %s", len(papers), dir_path)
    return papers


def _extract_title(text: str, fallback: str) -> str:
    """Best-effort title extraction from the first line."""
    first_line = text.strip().split("\n", 1)[0].strip()
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    if len(first_line) < 250:
        return first_line
    return fallback
