"""Step 1 — Extract the text of every corpus: unfiltered, then each condition's kept papers.

Writes ``retrieval_rag/corpora/<corpus>_corpus.json`` as
{paper_id: {"title", "full_text"}}: PyMuPDF text of every page, pages joined
by a blank line; the title is the first plausible line of the first page.
PDFs with no text are skipped with a warning. Existing corpus files are kept.

    python evaluation/retrieval_rag/build_corpora.py --settings <file>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import UNFILTERED, corpora_names, corpus_file  # noqa: E402
from evaluation.shared.decisions import kept_pdfs  # noqa: E402
from evaluation.shared.settings import add_settings_arg, load_settings  # noqa: E402

_META = ("doi:", "issn", "volume", "received", "accepted", "published")


def title_of(text: str, fallback: str) -> str:
    """First line of 11-299 characters that is not a URL or journal metadata."""
    for line in text.strip().split("\n")[:10]:
        line = line.strip()
        if 10 < len(line) < 300 and not line.startswith("http") and not any(k in line.lower() for k in _META):
            return line
    return fallback


def extract(pdfs: list) -> dict:
    import fitz
    from tqdm import tqdm

    corpus = {}
    for pdf in tqdm(pdfs, unit="pdf"):
        try:
            doc = fitz.open(str(pdf))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as exc:  # noqa: BLE001 — report and continue
            tqdm.write(f"  Warning: could not read {pdf.name}: {exc}")
            continue
        if not text.strip():
            tqdm.write(f"  Warning: no text in {pdf.name} (scanned PDF?) — skipped")
            continue
        corpus[pdf.stem] = {"title": title_of(text, pdf.stem), "full_text": text}
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    for corpus in corpora_names(s):
        out = corpus_file(s, corpus)
        if out.exists():
            print(f"  {corpus:<11} exists — kept ({out.name})")
            continue
        pdfs = sorted(s.papers_dir.glob("*.pdf")) if corpus == UNFILTERED else kept_pdfs(s.condition_dir(corpus))
        if not pdfs:
            print(f"  {corpus:<11} no PDFs found — skipped")
            continue
        data = extract(pdfs)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {corpus:<11} {len(data)} papers -> {out}")
    print("Next: start Chroma (chroma run --path runs/chroma --port 8000), then index_corpora.py")


if __name__ == "__main__":
    main()
