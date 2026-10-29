"""Phase 3: generate candidate eval questions from the corpus, for hand review.

Writes data/eval_candidates.json. Review it, then save the kept items as data/eval_set.json
(a separate file, so re-running this script never overwrites your reviewed set).
"""
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

CANDIDATES_PATH = config.DATA / "eval_candidates.json"
PER_COMPANY = {"text": 11, "table": 5}

BOILERPLATE = re.compile(
    r"exhibit index|incorporated by reference|exhibit description|power of attorney|"
    r"securities registered pursuant|table of contents|forward-looking statements|"
    r"pursuant to the requirements of section 13|signatures?\n", re.I)

SINGLE_PROMPT = """You are building an evaluation set for a retrieval system over SEC 10-K filings.
Read the {kind} below from {company_name}'s Form 10-K for fiscal year {fy} ({page_label}).

Write ONE question that:
- can be answered ONLY from this {kind}, with a specific fact (number, name, date, specific risk);
- names the company and, for financial figures, the fiscal year;
- is self-contained: never say "the passage", "the table", or "according to the text";
- sounds like something an analyst would actually ask.
{table_rule}
Also write the correct, concise answer (include units such as "$ millions").

If the {kind} is boilerplate (legal notices, signatures, lists of exhibits, table of contents)
or has no concrete fact, return {{"skip": true}}.

{kind_title}:
\"\"\"{text}\"\"\"

Return only JSON: {{"question": "...", "answer": "..."}}"""

TABLE_RULE = ("- for this table, the answer must require matching a specific row AND column "
              "(e.g. a line item for a specific year), not just reading a title.")

MULTI_PROMPT = """You are building an evaluation set for a retrieval system over SEC 10-K filings.
Below are excerpts from two companies' 10-K filings on the same topic.

Write ONE comparison question that needs BOTH excerpts to answer (e.g. which company is larger
on a metric, or the difference between them). Name both companies and their fiscal years.
Never say "the excerpt". Then write the correct answer, citing the figure from each company.
If the excerpts do not contain comparable facts, return {{"skip": true}}.

Excerpt A — {name_a} Form 10-K FY{fy_a}:
\"\"\"{text_a}\"\"\"

Excerpt B — {name_b} Form 10-K FY{fy_b}:
\"\"\"{text_b}\"\"\"

Return only JSON: {{"question": "...", "answer": "..."}}"""

# topic -> (chunk type, regex the chunk must match); used to pick comparable chunks per company
MULTI_TOPICS = {
    # the statement itself: title, then a "Net income" row; skips index tables and % tables
    "income statement": ("table", r"(statements? of (operations|income)|income statements)"
                                  r"(?![\s\S]*percentage)[\s\S]*\|\s*net income\s*\|"),
    "balance sheet": ("table", r"balance sheets?[\s\S]*total assets"),
    "employees": ("text", r"approximately [\d,]+ (full-time (equivalent )?)?(employees|people)"
                          r"|employed approximately"),
    "share repurchases": ("text", r"repurchased? [\s\S]{0,80}shares"),
}


def body(chunk) -> str:
    return chunk.page_content.split("\n", 1)[1]   # drop the "Company (TICK) Form 10-K..." header


def page_ref(chunk) -> dict:
    m = chunk.metadata
    return {"source": m["source"], "page": m["page"], "page_end": m["page_end"]}


def page_label(m: dict) -> str:
    return f"pages {m['page']}-{m['page_end']}" if m["page_end"] != m["page"] else f"page {m['page']}"


def eligible(chunk) -> bool:
    text = body(chunk)
    if BOILERPLATE.search(text):
        return False
    if chunk.metadata["type"] == "table":
        rows = [l for l in text.splitlines() if l.startswith("|") and re.search(r"\d", l)]
        return len(rows) >= 4
    return len(text) >= 400


def parse_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def ask(llm, prompt: str, retries: int = 4):
    for attempt in range(retries):
        try:
            return parse_json(llm.invoke(prompt).content)
        except Exception as e:   # rate limits, malformed JSON
            wait = 15 * (attempt + 1)
            print(f"    retry in {wait}s: {str(e)[:100]}")
            time.sleep(wait)
    return None


def single_hop(llm, chunks, rng, companies, used=frozenset()) -> list[dict]:
    items = []
    for ticker in companies:
        for kind, n in PER_COMPANY.items():
            pool = [c for c in chunks if c.metadata["company"] == ticker
                    and c.metadata["type"] == kind and eligible(c)
                    and c.metadata["chunk_id"] not in used]
            for c in rng.sample(pool, min(n, len(pool))):
                m = c.metadata
                out = ask(llm, SINGLE_PROMPT.format(
                    kind="table" if kind == "table" else "passage",
                    kind_title="Table" if kind == "table" else "Passage",
                    company_name=m["company_name"], fy=m["fiscal_year"], page_label=page_label(m),
                    table_rule=TABLE_RULE if kind == "table" else "", text=body(c)))
                if not out or out.get("skip") or not out.get("question") or not out.get("answer"):
                    print(f"  skip {m['chunk_id']}")
                    continue
                items.append({
                    "question": out["question"], "reference": out["answer"],
                    "reference_contexts": [c.page_content], "chunk_ids": [m["chunk_id"]],
                    "pages": [page_ref(c)], "companies": [ticker],
                    "type": "single_hop", "content_type": kind,
                })
                print(f"  {m['chunk_id']}: {out['question'][:90]}")
                time.sleep(1)
    return items


def multi_hop(llm, chunks, rng) -> list[dict]:
    items = []
    tickers = list(config.COMPANIES)
    for topic, (kind, pattern) in MULTI_TOPICS.items():
        rx = re.compile(pattern, re.I)
        first = {}
        for t in tickers:
            hits = [c for c in chunks if c.metadata["company"] == t and c.metadata["type"] == kind
                    and rx.search(body(c))]
            if hits:
                first[t] = hits[0]
        pairs = [(a, b) for i, a in enumerate(first) for b in list(first)[i + 1:]]
        for a, b in rng.sample(pairs, min(2, len(pairs))):
            ca, cb = first[a], first[b]
            out = ask(llm, MULTI_PROMPT.format(
                name_a=ca.metadata["company_name"], fy_a=ca.metadata["fiscal_year"], text_a=body(ca),
                name_b=cb.metadata["company_name"], fy_b=cb.metadata["fiscal_year"], text_b=body(cb)))
            if not out or out.get("skip") or not out.get("question") or not out.get("answer"):
                print(f"  skip multi-hop {topic} {a}/{b}")
                continue
            items.append({
                "question": out["question"], "reference": out["answer"],
                "reference_contexts": [ca.page_content, cb.page_content],
                "chunk_ids": [ca.metadata["chunk_id"], cb.metadata["chunk_id"]],
                "pages": [page_ref(ca), page_ref(cb)], "companies": [a, b],
                "type": "multi_hop", "content_type": kind, "topic": topic,
            })
            print(f"  multi-hop {topic} {a}/{b}: {out['question'][:80]}")
            time.sleep(1)
    return items


def main(seed: int, companies: list[str], append: bool):
    with open(config.CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    rng = random.Random(seed)
    llm = config.get_llm(reasoning_effort="low")

    existing = []
    if append and CANDIDATES_PATH.exists():
        existing = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    used = {cid for it in existing for cid in it["chunk_ids"]}

    print("Single-hop:")
    new = single_hop(llm, chunks, rng, companies, used)
    if not append:
        print("Multi-hop:")
        new += multi_hop(llm, chunks, rng)
    items = existing + new
    for i, it in enumerate(new, len(existing) + 1):
        it["id"] = f"q{i:03d}"

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(items)} candidates -> {CANDIDATES_PATH}")
    print(f"Review them, then save the kept items to {config.EVAL_SET_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--companies", nargs="+", default=list(config.COMPANIES),
                   help="tickers to sample single-hop questions from")
    p.add_argument("--append", action="store_true",
                   help="top up existing candidates (skips used chunks and multi-hop)")
    a = p.parse_args()
    main(a.seed, [t.upper() for t in a.companies], a.append)
