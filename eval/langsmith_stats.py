"""Phase 8: latency, tokens, cost and per-node timing of agent runs, from LangSmith traces.
"""
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)   # list_runs -> runs.query (2027)

import numpy as np
from langsmith import Client

import config
from agent.graph import RUN_NAME
from agent.tools import TOOLS

ROOT_NAMES = [RUN_NAME, "LangGraph"]
# USD per 1M tokens (input, output); fallback when LangSmith has no price. Checked 2026-10.
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "openai/gpt-oss-120b": (0.15, 0.60)}


def pct(values, q):
    return round(float(np.percentile(values, q)), 2) if values else None


def mean(values, ndigits=2):
    return round(float(np.mean(values)), ndigits) if values else None


def secs(run) -> float:
    return (run.end_time - run.start_time).total_seconds()


def main(days: int):
    client = Client()
    project = os.getenv("LANGSMITH_PROJECT", "agentic-rag")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    names = json.dumps(ROOT_NAMES)

    roots = [r for r in client.list_runs(project_name=project, is_root=True, start_time=since,
                                         filter=f"in(name, {names})") if r.end_time]
    ok = {r.trace_id: r for r in roots if not r.error}
    print(f"{len(roots)} agent runs in the last {days} days ({len(roots) - len(ok)} errored, excluded)")

    # child runs, fetched once per type and grouped by trace
    llm_runs, node_time, tool_time = defaultdict(list), defaultdict(dict), defaultdict(list)
    for r in client.list_runs(project_name=project, run_type="llm", start_time=since):
        if r.trace_id in ok and r.end_time:
            llm_runs[r.trace_id].append(r)
    for r in client.list_runs(project_name=project, start_time=since,
                              filter='and(eq(run_type, "chain"), in(name, ["agent", "tools"]))'):
        if r.trace_id in ok and r.end_time:
            node_time[r.trace_id][r.name] = node_time[r.trace_id].get(r.name, 0.0) + secs(r)
    tool_names = {t.name for t in TOOLS}   # skips nested runs such as tavily_search
    for r in client.list_runs(project_name=project, run_type="tool", start_time=since):
        if r.trace_id in ok and r.end_time and r.name in tool_names:
            tool_time[r.name].append(secs(r))

    rows = []
    for tid, root in ok.items():
        llms = llm_runs.get(tid, [])
        model = ((root.extra or {}).get("metadata", {}).get("model")
                 or next(((l.extra or {}).get("metadata", {}).get("ls_model_name") for l in llms), None))
        cost = float(root.total_cost) if root.total_cost else None
        if cost is None and model in PRICES:
            pin, pout = PRICES[model]
            cost = sum((l.prompt_tokens or 0) * pin + (l.completion_tokens or 0) * pout
                       for l in llms) / 1e6
        rows.append({"model": model, "source": (root.tags or ["untagged"])[0],
                     "latency_s": secs(root), "tokens": root.total_tokens, "cost": cost,
                     "llm_calls": len(llms), **{f"{k}_s": v for k, v in node_time[tid].items()}})

    def summarize(rs):
        lat = [r["latency_s"] for r in rs]
        node_means = {n: mean([r.get(f"{n}_s", 0.0) for r in rs]) for n in ("agent", "tools")}
        return {
            "runs": len(rs), "p50_latency_s": pct(lat, 50), "p95_latency_s": pct(lat, 95),
            "mean_latency_s": mean(lat),
            "avg_tokens_per_query": mean([r["tokens"] for r in rs if r["tokens"]], 0),
            "avg_cost_per_query_usd": mean([r["cost"] for r in rs if r["cost"] is not None], 5),
            "avg_llm_calls_per_query": mean([r["llm_calls"] for r in rs]),
            "avg_node_time_s": node_means,
            "slowest_node": max(node_means, key=lambda n: node_means[n] or 0),
        }

    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    headline = by_model.get(config.LLM_MODEL) or rows
    stats = {
        "model": config.LLM_MODEL, "days": days, **summarize(headline),
        "by_source": {s: summarize([r for r in headline if r["source"] == s])
                      for s in sorted({r["source"] for r in headline})},
        "by_model": {str(m): summarize(rs) for m, rs in by_model.items()},
        "tool_latency_s": {t: {"calls": len(v), "p50": pct(v, 50), "p95": pct(v, 95)}
                           for t, v in sorted(tool_time.items())},
    }
    (config.RESULTS_DIR / "langsmith_stats.json").write_text(json.dumps(stats, indent=2),
                                                             encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    main(p.parse_args().days)
