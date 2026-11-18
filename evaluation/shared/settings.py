"""Evaluation settings.

Everything that defines an evaluation run lives in one YAML file: where the
PROSPECT run folder is, where outputs go, which models play which role, who
the human raters are, sample sizes and seeds. ``evaluation/settings.yaml`` is
the template; ``evaluation/examples/scenario_1/settings.yaml`` reproduces the
paper's configuration.

Relative paths in the settings file are resolved against the repository root,
so run every script from there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

EVAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVAL_ROOT.parent
DEFAULT_SETTINGS = EVAL_ROOT / "settings.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MISSING = object()


class SettingsError(SystemExit):
    """A setting is missing or points to something that does not exist."""


class Settings:
    """Read-only view of the settings file, with path helpers."""

    def __init__(self, data: Dict[str, Any], source: Path) -> None:
        self.data = data
        self.source = source

    # ── generic access ──────────────────────────────────────────────────────

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Return a value by dotted key, e.g. ``get("filter_validity.sample.seed")``."""
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is not _MISSING:
                return default
            else:
                raise SettingsError(f"\nERROR: setting '{dotted}' is missing in {self.source}.\n")
        return node

    def resolve(self, value: Optional[str]) -> Optional[Path]:
        """Resolve a path from the settings file against the repository root."""
        if value in (None, ""):
            return None
        p = Path(value)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    def path(self, dotted: str, default: Any = _MISSING) -> Optional[Path]:
        return self.resolve(self.get(dotted, default))

    # ── the PROSPECT run being evaluated ────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        p = self.path("run_dir")
        if not p.is_dir():
            raise SettingsError(f"\nERROR: run_dir does not exist: {p}\n")
        return p

    @property
    def conditions(self) -> List[str]:
        """The filter conditions to evaluate, in the order they are reported."""
        return list(self.get("conditions"))

    def condition_dir(self, condition: str) -> Path:
        """Output folder of one condition (default ``<run_dir>/<condition>``)."""
        override = (self.get("condition_dirs", {}) or {}).get(condition)
        return self.resolve(override) if override else self.run_dir / condition

    @property
    def papers_dir(self) -> Path:
        """The PDFs that were filtered: setting, else run_inputs.json, else <run_dir>/papers."""
        explicit = self.path("papers_dir", None)
        if explicit:
            return explicit
        lock = self.run_dir / "run_inputs.json"
        if lock.exists():
            recorded = json.loads(lock.read_text(encoding="utf-8")).get("papers_dir")
            if recorded:
                p = Path(recorded)
                return p if p.is_absolute() else (self.run_dir / p).resolve()
        return self.run_dir / "papers"

    def operational_config(self) -> Dict[str, Any]:
        """The run's operational_config.json (criteria, queries, application description)."""
        p = self.path("operational_config", None) or self.run_dir / "operational_config.json"
        if not p.exists():
            raise SettingsError(f"\nERROR: operational config not found: {p}\n")
        return json.loads(p.read_text(encoding="utf-8"))

    def application_description(self) -> str:
        """Text shown to every rater: a file from the settings, else the operational config's."""
        p = self.path("application_description_file", None)
        if p:
            return p.read_text(encoding="utf-8").strip()
        return self.operational_config()["application_description"].strip()

    # ── where outputs go ────────────────────────────────────────────────────

    @property
    def workspace(self) -> Path:
        return self.path("workspace")

    def out(self, *parts: str) -> Path:
        """A path inside the workspace; parent folders are created."""
        p = self.workspace.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def add_settings_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--settings", default=str(DEFAULT_SETTINGS),
        help="Evaluation settings file (default: evaluation/settings.yaml)",
    )


def load_settings(path: Optional[str] = None) -> Settings:
    """Load the settings file and the repository's .env (keys are read by name only)."""
    from common.env import load_env

    load_env()
    p = Path(path) if path else DEFAULT_SETTINGS
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise SettingsError(f"\nERROR: settings file not found: {p}\n")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    todo = list(_todo_keys(data))
    if todo:
        print(f"Warning: {p.name} still has TODO values: {', '.join(todo[:6])}"
              + (" ..." if len(todo) > 6 else ""))
    return Settings(data, p)


def _todo_keys(node: Any, prefix: str = ""):
    """Dotted keys whose value still starts with "TODO"."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _todo_keys(v, f"{prefix}{k}.")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _todo_keys(v, f"{prefix}{i}.")
    elif isinstance(node, str) and node.strip().startswith("TODO"):
        yield prefix.rstrip(".")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def to_markdown(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """A dependency-free markdown table."""
    if not rows:
        return "_none_"
    cols = columns or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
