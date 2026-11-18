"""Read keep/discard decisions from a PROSPECT run folder.

Each condition folder holds ``paper_decisions/*.json`` (one file per paper) and
``routed_pdfs/keep/`` (copies of the kept PDFs). A paper is identified by its
PDF file name without ``.pdf`` (the "paper ID").
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")


def paper_id_of(decision_file: Path, data: Dict) -> str:
    """Paper ID of one decision file.

    Prefers an explicit ``paper_id``; then the PDF name recorded by the A3
    family (``run_metrics.pdf``) or by A1/A2 (``pdf``); then the file name
    without the ``_YYYYMMDD_HHMMSS`` timestamp the A3 family appends.
    """
    if data.get("paper_id"):
        return data["paper_id"]
    pdf = (data.get("run_metrics") or {}).get("pdf") or data.get("pdf")
    if pdf:
        return Path(pdf).stem
    return _TIMESTAMP.sub("", decision_file.stem)


def load_decisions(condition_dir: Path) -> Dict[str, str]:
    """Return {paper_id: "keep" | "discard"} for one condition."""
    folder = condition_dir / "paper_decisions"
    if not folder.is_dir():
        raise FileNotFoundError(f"No paper_decisions/ folder in {condition_dir}")
    out: Dict[str, str] = {}
    for f in sorted(folder.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[paper_id_of(f, data)] = str(data.get("decision", "discard")).lower()
    return out


def kept_pdfs(condition_dir: Path) -> List[Path]:
    """The kept PDFs of one condition, in name order."""
    return sorted((condition_dir / "routed_pdfs" / "keep").glob("*.pdf"))
