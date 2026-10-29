"""Phase 7: tool-routing accuracy on data/routing_set.json.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from agent.graph import STEP_LIMIT_MSG, ask, build_graph, tool_calls


def main(tag: str, limit: int | None):
    items = json.loads(config.ROUTING_SET_PATH.read_text(encoding="utf-8"))[:limit]
    graph = build_graph()   # no checkpointer: each question is independent
    expected_n, called_n, hits = Counter(), Counter(), Counter()
    rows = []

    for it in items:
        t0 = time.perf_counter()
        try:
            turn = ask(graph, it["question"], source="routing-eval")
            called, answer, error = [tc["name"] for tc in tool_calls(turn)], turn[-1].content, None
        except Exception as e:
            called, answer, error = [], None, str(e)[:300]
        secs = time.perf_counter() - t0

        exp, got = set(it["expected_tools"]), set(called)
        expected_n.update(exp)
        called_n.update(got)
        hits.update(exp & got)
        rows.append({**it, "called": called, "exact": exp == got, "coverage": exp <= got,
                     "latency_s": round(secs, 2), "answer": answer, "error": error})
        mark = "OK " if exp == got else ("~  " if exp <= got else "X  ")
        print(f"{mark}{it['question'][:70]:70} expected {sorted(exp)} got {called}"
              + (f"  ERROR {error[:80]}" if error else ""))

    n = len(rows)
    tools = sorted(set(expected_n) | set(called_n))
    summary = {
        "tag": tag, "n": n, "model": config.LLM_MODEL,
        "exact_match_acc": round(sum(r["exact"] for r in rows) / n, 3),
        "coverage_acc": round(sum(r["coverage"] for r in rows) / n, 3),
        "errors": sum(r["error"] is not None for r in rows),
        # right tools but no answer: the agent looped until the step limit
        "step_limit_hits": sum(r["answer"] == STEP_LIMIT_MSG for r in rows),
        "per_tool": {t: {"recall": round(hits[t] / expected_n[t], 3) if expected_n[t] else None,
                         "precision": round(hits[t] / called_n[t], 3) if called_n[t] else None,
                         "expected": expected_n[t], "called": called_n[t]} for t in tools},
    }
    out = config.RESULTS_DIR / f"routing_{tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="before", help="results file suffix, e.g. before / after")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    main(a.tag, a.limit)
