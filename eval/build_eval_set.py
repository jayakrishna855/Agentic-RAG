"""Phase 3: apply the hand-review decisions to data/eval_candidates.json, merge the
hand-written questions in data/eval_manual.json, and write data/eval_set.json.

Keeping review decisions in code makes the eval set reproducible and auditable:
every dropped or edited item has a written reason.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config

CANDIDATES_PATH = config.DATA / "eval_candidates.json"
MANUAL_PATH = config.DATA / "eval_manual.json"
REQUIRED = ("question", "reference", "companies", "pages", "type", "content_type")

DROP = {
    "q002": "vague: answer is just a reporting date",
    "q003": "governance boilerplate (board vacancy rules)",
    "q014": "boilerplate: every 10-K says controls are effective",
    "q020": "duplicate of q013",
    "q035": "table-of-contents trivia (page number)",
    "q045": "boilerplate (Delaware anti-takeover law)",
    "q046": "trivial (fiscal year-end date)",
    "q053": "exhibit description of debt notes, not analyst-relevant",
    "q054": "exhibit description of securities (governance trivia)",
    "q055": "exhibit trivia (S-8 registration number)",
    "q056": "exhibit description of debt notes, not analyst-relevant",
    "q057": "signature-page trivia",
    "q061": "signature-page trivia",
}

FIXES = {
    "q037": {
        "question": "How much owned square footage did Amazon report for AWS as of December 31, 2025?",
        "reference": "28,469 thousand square feet (about 28.5 million square feet)",
        "_reason": "table is in thousands; generated answer dropped the unit",
    },
}


def load_manual() -> list[dict]:
    if not MANUAL_PATH.exists():
        return []
    sources = {p.name for p in config.RAW_DIR.glob("*.pdf")}
    items = []
    for n, it in enumerate(json.loads(MANUAL_PATH.read_text(encoding="utf-8")), 1):
        iid = f"m{n:03d}"
        missing = [k for k in REQUIRED if not it.get(k)]
        assert not missing, f"{iid}: missing fields {missing}"
        assert set(it["companies"]) <= set(config.COMPANIES), f"{iid}: unknown company"
        for p in it["pages"]:
            assert p["source"] in sources, f"{iid}: unknown source {p['source']}"
            assert 1 <= p["page"] <= p["page_end"], f"{iid}: bad page range {p}"
        items.append({"reference_contexts": [], "chunk_ids": [], **it,
                      "id": iid, "origin": "manual", "reviewed": True})
    return items


def main():
    items = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    unknown = (set(DROP) | set(FIXES)) - {it["id"] for it in items}
    assert not unknown, f"review refers to unknown ids: {unknown}"

    kept = []
    for it in items:
        if it["id"] in DROP:
            continue
        fix = {k: v for k, v in FIXES.get(it["id"], {}).items() if not k.startswith("_")}
        kept.append({**it, **fix, "origin": "generated", "reviewed": True})
    n_generated = len(kept)
    kept += load_manual()

    config.EVAL_SET_PATH.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    by = Counter((it["type"], it["content_type"]) for it in kept)
    per_company = Counter(c for it in kept for c in it["companies"])
    print(f"{len(items)} candidates -> {n_generated} kept ({len(DROP)} dropped, {len(FIXES)} fixed)"
          f" + {len(kept) - n_generated} manual = {len(kept)} total")
    print("by type:", dict(by))
    print("per company (multi-hop counts for both):", dict(per_company))


if __name__ == "__main__":
    main()
