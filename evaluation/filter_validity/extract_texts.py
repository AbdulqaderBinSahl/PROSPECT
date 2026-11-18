"""Step 2 — Extract each paper's text for the LLM judges (reference list removed).

Writes ``filter_validity/texts/<paper_id>.txt`` for every PDF in the run's
paper folder: PyMuPDF text of every page, pages joined by a blank line. The
reference list is cut from the last "References"-type heading found in the
final 40% of the lines. Existing files are kept (safe to re-run).

    python evaluation/filter_validity/extract_texts.py --settings <file>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.filter_validity.panel import texts_dir  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402

_REF_HEADING = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|acknowledgements?\s+and\s+references)\s*$",
    re.IGNORECASE,
)


def extract_text(pdf_path: Path) -> str:
    """Text of every non-empty page, stripped, joined by a blank line."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages = [t for t in (page.get_text("text").strip() for page in doc) if t]
    doc.close()
    return "\n\n".join(pages)


def strip_references(text: str) -> tuple[str, bool]:
    """Remove everything from the last reference heading in the final 40% of lines."""
    lines = text.splitlines()
    cutoff = int(len(lines) * 0.60)
    last = None
    for i, line in enumerate(lines[cutoff:], cutoff):
        if _REF_HEADING.match(line):
            last = i
    if last is None:
        return text, False
    return "\n".join(lines[:last]).rstrip() + "\n", True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)
    strip = bool(s.get("filter_validity.texts.strip_references", True))

    pdfs = sorted(s.papers_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"\nERROR: no PDFs in {s.papers_dir}\n")
    out_dir = texts_dir(s)
    from tqdm import tqdm

    done = skipped = failed = stripped = 0
    for pdf in tqdm(pdfs, unit="pdf"):
        out = out_dir / f"{pdf.stem}.txt"
        if out.exists():
            skipped += 1
            continue
        try:
            text = extract_text(pdf)
        except Exception as exc:  # noqa: BLE001 — corrupt or protected PDF
            tqdm.write(f"  Warning: could not read {pdf.name}: {exc}")
            failed += 1
            continue
        if strip:
            text, changed = strip_references(text)
            stripped += changed
        out.write_text(text, encoding="utf-8")
        done += 1
    print(f"\nExtracted {done}, kept existing {skipped}, failed {failed}; "
          f"reference lists removed from {stripped}. Texts: {out_dir}")
    print("Next: python evaluation/filter_validity/draw_sample.py")


if __name__ == "__main__":
    main()
