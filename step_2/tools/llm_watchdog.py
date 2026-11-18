"""
Supervise a long PROSPECT Step 2 run against a local LM Studio server (optional).

Some local models degenerate after many hours of continuous inference: they
start replying with one character repeated to the token limit, and every later
request on the poisoned slot fails the same way. This watchdog detects that,
reloads the model through LM Studio's `lms` CLI, and relaunches
``run_step2.py --resume`` (already-decided papers are skipped).

Detection uses the pipeline's own bad_outputs/ folder (written by A3, A3w and
A4), so it adds no load to the server: a file whose content has <= 2 distinct
characters is a degenerate reply. A stalled run (no new decision for a long
time) triggers the same recovery, as does the pipeline exiting non-zero.

The run folder must already be set up: either Step 1 output exists in
<run-dir>/step1/, or you ran ``run_step2.py --prepare-only`` with --input /
--papers-dir once.

Usage (from the repository root):
    python step_2/tools/llm_watchdog.py --run-dir runs/my_topic
    python step_2/tools/llm_watchdog.py --run-dir runs/my_topic --condition A3 --max-restarts 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]  # repository root
PYTHON = sys.executable
LMS = Path.home() / ".lmstudio" / "bin" / "lms.exe"
if not LMS.exists():
    LMS = Path.home() / ".lmstudio" / "bin" / "lms"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Watchdog for a long Step 2 run against LM Studio")
    p.add_argument("--run-dir", type=Path, required=True, help="Run folder passed to run_step2.py")
    p.add_argument("--condition", default="A3w", help="Condition passed to run_step2.py (default: A3w)")
    p.add_argument("--model", default="openai/gpt-oss-120b", help="Model to (re)load in LM Studio")
    p.add_argument("--context-length", type=int, default=32768)
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--endpoint", default="http://127.0.0.1:1234/v1")
    p.add_argument("--check-interval", type=int, default=120, help="Seconds between health checks (default: 120)")
    p.add_argument("--degenerate-window", type=int, default=900, help="Look-back window in seconds for degenerate replies (default: 900)")
    p.add_argument("--degenerate-threshold", type=int, default=3, help="Degenerate replies in the window before recovering (default: 3)")
    p.add_argument("--stall-minutes", type=int, default=90, help="Recover if no new decision for this long (default: 90)")
    p.add_argument("--max-restarts", type=int, default=10)
    return p.parse_args()


def log(message: str, logfile: Path) -> None:
    """Print and append a timestamped line to the watchdog log."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    with logfile.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def degenerate_count(bad_dir: Path, window_sec: int, launched_at: float) -> int:
    """Number of degenerate model replies produced by the current pipeline run.

    Bounded by the launch time as well as the window: files written before this
    run started must not re-trigger recovery after a relaunch.
    """
    if not bad_dir.is_dir():
        return 0
    cutoff = max(time.time() - window_sec, launched_at)
    count = 0
    for f in bad_dir.glob("*.txt"):
        if f.stat().st_mtime < cutoff:
            continue
        try:
            if len(set(f.read_text(encoding="utf-8", errors="replace"))) <= 2:
                count += 1
        except OSError:
            continue
    return count


def minutes_without_progress(decisions_dir: Path, launched_at: float) -> float:
    """Minutes since this pipeline run last made progress.

    Measured from the launch time or the newest decision, whichever is later:
    decisions written by earlier runs must not count as a stall.
    """
    files = list(decisions_dir.glob("*.json")) if decisions_dir.is_dir() else []
    newest = max((f.stat().st_mtime for f in files), default=0.0)
    return (time.time() - max(launched_at, newest)) / 60


def server_healthy(endpoint: str, model: str) -> tuple[bool, str]:
    """Probe the endpoint with a tiny prompt. Returns (healthy, reason)."""
    try:
        requests.get(f"{endpoint}/models", timeout=10).raise_for_status()
    except Exception as exc:
        return False, f"models endpoint unreachable ({type(exc).__name__})"
    try:
        r = requests.post(
            f"{endpoint}/chat/completions",
            timeout=180,
            json={"model": model, "messages": [{"role": "user", "content": 'Return only: {"ok": 1}'}],
                  "temperature": 0.1, "max_tokens": 40},
        )
        if not r.ok:
            return False, f"chat completions HTTP {r.status_code}"
        text = r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return False, f"chat completions failed ({type(exc).__name__})"
    if len(set(text)) <= 2:
        return False, "degenerate reply to probe"
    return True, "ok"


def reload_model(args: argparse.Namespace, logfile: Path) -> bool:
    """Unload and reload the model with the configured context and parallelism."""
    if not LMS.exists():
        log(f"  lms CLI not found at {LMS}; cannot reload automatically", logfile)
        return False
    for cmd in (
        [str(LMS), "unload", "--all"],
        [str(LMS), "load", args.model, "--context-length", str(args.context_length),
         "--parallel", str(args.parallel), "--gpu", "max", "-y"],
    ):
        log(f"  running: {' '.join(cmd[1:])}", logfile)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            log(f"  FAILED ({result.returncode}): {(result.stderr or result.stdout)[:300]}", logfile)
            return False
    healthy, reason = server_healthy(args.endpoint, args.model)
    log(f"  post-reload probe: {reason}", logfile)
    return healthy


def launch_pipeline(args: argparse.Namespace, run_log: Path) -> subprocess.Popen:
    """Start run_step2.py with --resume, appending stdout/stderr to the run log."""
    cmd = [
        PYTHON, str(ROOT / "step_2" / "run_step2.py"),
        "--run-dir", str(args.run_dir),
        "--condition", args.condition,
        "--base-url", args.endpoint,
        "--model", args.model,
        "--resume",
    ]
    handle = run_log.open("a", encoding="utf-8")
    env_python_unbuffered = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env_python_unbuffered}
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))


def main() -> None:
    """Run the pipeline under supervision, restarting it when the model degenerates."""
    args = parse_args()
    sys.path.insert(0, str(ROOT / "step_2"))
    from rag_corpus_filter.run_setup import canonical_condition

    args.condition = canonical_condition(args.condition)
    out = args.run_dir.resolve() / args.condition
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "watchdog.log"
    run_log = out / "run.log"
    bad_dir = out / "bad_outputs"
    decisions_dir = out / "paper_decisions"

    log(f"watchdog start | model={args.model} context={args.context_length} parallel={args.parallel}", logfile)
    healthy, reason = server_healthy(args.endpoint, args.model)
    log(f"initial probe: {reason}", logfile)
    if not healthy and not reload_model(args, logfile):
        log("cannot reach a healthy model; aborting", logfile)
        sys.exit(1)

    restarts = 0
    launched_at = time.time()
    proc = launch_pipeline(args, run_log)
    log(f"pipeline started (pid {proc.pid})", logfile)

    while True:
        time.sleep(args.check_interval)
        exit_code = proc.poll()

        if exit_code == 0:
            log("pipeline finished successfully", logfile)
            n = len(list(decisions_dir.glob("*.json")))
            log(f"decisions on disk: {n}", logfile)
            return

        trigger = None
        if exit_code is not None:
            trigger = f"pipeline exited with code {exit_code}"
        else:
            n_degenerate = degenerate_count(bad_dir, args.degenerate_window, launched_at)
            stalled = minutes_without_progress(decisions_dir, launched_at)
            if n_degenerate >= args.degenerate_threshold:
                trigger = f"{n_degenerate} degenerate replies in the last {args.degenerate_window // 60} min"
            elif stalled >= args.stall_minutes:
                trigger = f"no new decision for {stalled:.0f} min"

        if not trigger:
            continue

        log(f"RECOVERY TRIGGERED: {trigger}", logfile)
        restarts += 1
        if restarts > args.max_restarts:
            log(f"max restarts ({args.max_restarts}) reached; stopping", logfile)
            if proc.poll() is None:
                proc.terminate()
            sys.exit(1)

        if proc.poll() is None:
            log("  stopping pipeline", logfile)
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()

        if not reload_model(args, logfile):
            log("  model reload failed; retrying next cycle", logfile)
            time.sleep(args.check_interval)
            continue

        launched_at = time.time()
        proc = launch_pipeline(args, run_log)
        log(f"  pipeline relaunched (restart {restarts}/{args.max_restarts}, pid {proc.pid})", logfile)


if __name__ == "__main__":
    main()
