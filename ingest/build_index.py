"""Phase 1: PDFs -> markdown pages -> text + table chunks with headers -> Chroma + chunks.pkl.

Changes vs. the original plan:
- pymupdf4llm extracts each page as markdown, so tables keep their rows and columns.
- Running headers/footers repeated across pages are stripped.
- A table that continues onto the next page is merged into one logical table.
- Each table is kept whole in one chunk (with its title as context); tables longer than
  TABLE_MAX_CHARS are split by rows, repeating the header row in every piece.
- Each chunk is prefixed with a header ("Apple (AAPL) Form 10-K FY2025, page 23") so both
  BM25 and the embedding know which company a mid-document chunk belongs to.
- chunk_id is deterministic ("AAPL-p23-1"), and source+page range is stored so the eval can
  judge relevance by page, which survives re-chunking.
"""
import json
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

BATCH = 500   # Chroma rejects very large single inserts
SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")


# ---------- PDF -> cleaned markdown lines per page ----------

def to_markdown_pages(pdf: Path) -> list[str]:
    """pymupdf4llm is slow (~0.5 s/page), so cache its output per PDF."""
    config.MD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.MD_CACHE_DIR / f"{pdf.stem}.json"
    key = f"{pdf.stat().st_size}-{int(pdf.stat().st_mtime)}"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data["key"] == key:
            return data["pages"]
    pages = [c["text"] for c in pymupdf4llm.to_markdown(str(pdf), page_chunks=True)]
    cache.write_text(json.dumps({"key": key, "pages": pages}), encoding="utf-8")
    return pages


def clean_line(line: str) -> str:
    line = line.replace("\xa0", " ").replace("<br>", " ")
    line = re.sub(r"</?u>|\*\*", "", line)          # underline tags, bold markers
    return re.sub(r"[ \t]+", " ", line).strip()


def is_table(line: str) -> bool:
    return line.startswith("|")


def norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.lstrip("#").strip()).lower()


def boilerplate(pages: list[list[str]]) -> set[str]:
    """Non-table lines repeated on >=25% of pages (running headers, page footers)."""
    counts = Counter()
    for lines in pages:
        counts.update({norm(l) for l in lines if not is_table(l)})
    cutoff = max(3, 0.25 * len(pages))
    return {k for k, n in counts.items() if n >= cutoff and len(k) < 100}


# ---------- blocks: runs of text or table lines, tables merged across pages ----------

def ncols(line: str) -> int:
    return line.count("|") - 1


def page_blocks(lines: list[str], page: int) -> list[dict]:
    blocks = []
    for line in lines:
        kind = "table" if is_table(line) else "text"
        if blocks and blocks[-1]["kind"] == kind:
            blocks[-1]["lines"].append(line)
        else:
            blocks.append({"kind": kind, "lines": [line], "page": page, "page_end": page})
    return blocks


def continues(prev: dict, cur: dict) -> bool:
    """cur (first block on a page) is the rest of prev (last block on the previous page)."""
    return (prev["kind"] == cur["kind"] == "table"
            and prev["page_end"] == cur["page"] - 1
            and ncols(prev["lines"][0]) == ncols(cur["lines"][0]))


def merge_into(prev: dict, cur: dict):
    lines = cur["lines"]
    if len(lines) > 1 and lines[0] == prev["lines"][0] and SEPARATOR.match(lines[1]):
        rows = lines[2:]                                  # header repeated: drop it
    else:
        rows = [l for l in lines if not SEPARATOR.match(l)]   # headerless continuation
    prev["lines"].extend(rows)
    prev["page_end"] = cur["page"]


def doc_blocks(md_pages: list[str]) -> tuple[list[dict], int]:
    pages = [[clean_line(l) for l in md.splitlines()] for md in md_pages]
    pages = [[l for l in lines if l and l != "#"] for lines in pages]
    skip = boilerplate(pages)
    blocks, merges = [], 0
    for i, lines in enumerate(pages):
        lines = [l for l in lines if is_table(l) or norm(l) not in skip]
        for j, b in enumerate(page_blocks(lines, page=i + 1)):
            if j == 0 and blocks and continues(blocks[-1], b):
                merge_into(blocks[-1], b)
                merges += 1
            else:
                blocks.append(b)
    return blocks, merges


# ---------- blocks -> chunks ----------

def split_table(lines: list[str], context: str) -> list[str]:
    if len(lines) > 1 and SEPARATOR.match(lines[1]):
        head, rows = lines[:2], lines[2:]
    else:
        head, rows = [], lines
    prefix = "\n".join(([context] if context else []) + head)
    pieces, cur, size = [], [], len(prefix)
    for r in rows:
        if cur and size + len(r) + 1 > config.TABLE_MAX_CHARS:
            pieces.append(cur)
            cur, size = [], len(prefix)
        cur.append(r)
        size += len(r) + 1
    if cur:
        pieces.append(cur)
    return [f"{prefix}\n" + "\n".join(p) for p in pieces]


def table_context(prev_block: dict | None) -> str:
    """Title lines just above a table, e.g. 'CONSOLIDATED BALANCE SHEETS / (In millions)'."""
    if not prev_block or prev_block["kind"] != "text":
        return ""
    picked = []
    for line in reversed(prev_block["lines"][-3:]):
        if picked and sum(map(len, picked)) + len(line) > 300:
            break
        picked.insert(0, line)
    if len(picked) == 1 and len(picked[0]) > 300:     # one long paragraph: keep its tail
        picked[0] = picked[0][-300:].split(" ", 1)[-1]
    return "\n".join(picked)


def page_label(m: dict) -> str:
    if m["page_end"] != m["page"]:
        return f"pages {m['page']}-{m['page_end']}"
    return f"page {m['page']}"


def header(m: dict) -> str:
    return f"{m['company_name']} ({m['company']}) Form 10-K FY{m['fiscal_year']}, {page_label(m)}"


def chunk_pdf(pdf: Path, splitter) -> tuple[list[Document], int, int]:
    ticker, _, year = pdf.stem.split("_")   # AAPL_10K_2025
    if ticker not in config.COMPANIES:
        raise ValueError(f"{pdf.name}: unknown ticker {ticker}; add it to config.COMPANIES")
    base = {"source": pdf.name, "company": ticker,
            "company_name": config.COMPANIES[ticker], "fiscal_year": int(year)}

    md_pages = to_markdown_pages(pdf)
    blocks, merges = doc_blocks(md_pages)

    chunks, per_page = [], Counter()
    for k, b in enumerate(blocks):
        if b["kind"] == "table":
            texts = split_table(b["lines"], table_context(blocks[k - 1] if k else None))
        else:
            texts = splitter.split_text("\n".join(b["lines"]))
        for t in texts:
            if len(t) <= 100 or all(l.startswith("#") for l in t.splitlines()):
                continue                        # drop near-empty and heading-only chunks
            m = {**base, "page": b["page"], "page_end": b["page_end"], "type": b["kind"]}
            m["chunk_id"] = f"{ticker}-p{b['page']}-{per_page[b['page']]}"
            per_page[b["page"]] += 1
            chunks.append(Document(f"{header(m)}\n{t}", metadata=m))
    return chunks, len(md_pages), merges


def main():
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    assert pdfs, f"No PDFs found in {config.RAW_DIR}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n#", "\n\n", "\n", ". ", " ", ""],
    )
    chunks, n_pages, n_merges = [], 0, 0
    for pdf in pdfs:
        t0 = time.perf_counter()
        c, p, m = chunk_pdf(pdf, splitter)
        chunks += c
        n_pages += p
        n_merges += m
        print(f"  {pdf.name}: {p} pages -> {len(c)} chunks, "
              f"{m} cross-page table merges ({time.perf_counter() - t0:.0f}s)")

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    vs = Chroma(
        collection_name=config.COLLECTION,
        embedding_function=config.get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    t0 = time.perf_counter()
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vs.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        print(f"  embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")
    embed_secs = time.perf_counter() - t0

    with open(config.CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    lengths = [len(c.page_content) for c in chunks]
    stats = {
        "documents": len(pdfs),
        "pages": n_pages,
        "chunks": len(chunks),
        "text_chunks": sum(c.metadata["type"] == "text" for c in chunks),
        "table_chunks": sum(c.metadata["type"] == "table" for c in chunks),
        "cross_page_table_merges": n_merges,
        "chunks_per_company": {t: sum(c.metadata["company"] == t for c in chunks)
                               for t in config.COMPANIES},
        "avg_chunk_chars": round(sum(lengths) / len(lengths)),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "table_max_chars": config.TABLE_MAX_CHARS,
        "embed_model": config.EMBED_MODEL,
        "embed_seconds": round(embed_secs, 1),
    }
    with open(config.RESULTS_DIR / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
