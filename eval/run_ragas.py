"""Phase 4: retrieval ablation. Every config runs through the same answer prompt and judge.
"""
import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

import config
from retrieval.retriever import Retriever

CONFIGS = {   # name -> (retrieval mode, reranker model)
    "dense": ("dense", None),
    "hybrid": ("hybrid", None),
    "hybrid_rerank": ("hybrid_rerank", config.RERANK_MODEL),
    "hybrid_rerank_fast": ("hybrid_rerank", config.RERANK_MODEL_FAST),
}
CACHE = config.RESULTS_DIR / "cache"
ANSWERS = CACHE / "answers.jsonl"
JUDGE = CACHE / "judge.jsonl"
RETRIEVAL_METRICS = ["hit@5", "mrr", "ctx_precision", "page_recall"]
LLM_METRICS = ["context_recall", "faithfulness", "answer_relevancy"]
MAX_QUOTA_WAIT = 15 * 60   # wait out short rate-limit windows; stop on longer ones

ANSWER_PROMPT = """Answer the question using ONLY the context below. Each context starts with
its source (company, filing, page). Be concise and include units and fiscal years.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


class QuotaExhausted(Exception):
    pass


# ---------- retrieval ----------

def relevant(meta: dict, pages: list[dict]) -> bool:
    return any(meta["source"] == p["source"] and meta["page"] <= p["page_end"]
               and meta["page_end"] >= p["page"] for p in pages)


def retrieval_scores(metas: list[dict], pages: list[dict]) -> dict:
    rel = [relevant(m, pages) for m in metas]
    hits = [k for k, r in enumerate(rel, 1) if r]
    # RAGAS context precision: mean of precision@k over the ranks k that are relevant
    precision = (sum(sum(rel[:k]) / k for k in hits) / len(hits)) if hits else 0.0
    covered = sum(any(relevant(m, [p]) for m in metas) for p in pages)
    return {"hit@5": float(bool(hits)), "mrr": 1 / hits[0] if hits else 0.0,
            "ctx_precision": precision, "page_recall": covered / len(pages)}


def retrieve_all(items: list[dict], names: list[str], rerun: bool) -> dict[str, dict]:
    """Returns {config: {item id: row}}; cached per config because the bge reranker is slow."""
    out, retriever = {}, None
    for name in names:
        path = CACHE / f"retrieval_{name}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not rerun else {}
        todo = [it for it in items if it["id"] not in cached]
        if todo:
            if retriever is None:
                retriever = Retriever()
            mode, rerank_model = CONFIGS[name]
            if rerank_model:
                retriever.rerank_model = rerank_model
            retriever.warmup()   # load models so latency excludes loading
            print(f"retrieval {name}: {len(todo)} questions")
            for it in todo:
                docs, secs = retriever.timed_retrieve(it["question"], mode)
                cached[it["id"]] = {
                    "chunk_ids": [d.metadata["chunk_id"] for d in docs],
                    "contexts": [d.page_content for d in docs],
                    "metas": [{k: d.metadata[k] for k in ("source", "page", "page_end")} for d in docs],
                    "latency_s": secs,
                }
            path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        out[name] = cached
    return out


# ---------- caching + rate limits ----------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_key(question: str, chunk_ids: list[str]) -> str:
    """Same question + same retrieved chunks -> same answer and scores, whatever the config."""
    raw = f"{config.LLM_MODEL}|{question}|{','.join(sorted(chunk_ids))}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_scores() -> dict[str, dict]:
    scores = {}
    for r in load_jsonl(JUDGE):
        if r.get("judge") == config.JUDGE_MODEL:
            scores.setdefault(r["key"], {})[r["metric"]] = r["score"]
    return scores


def quota_wait(err: Exception) -> float | None:
    """Seconds to wait for a Groq rate limit, None if it isn't one; raises if it's too long."""
    msg = str(err)
    if "rate limit" not in msg.lower() and "429" not in msg:
        return None
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", msg)
    secs = 60.0
    if m and any(m.groups()):
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        secs = h * 3600 + mi * 60 + s
    if secs > MAX_QUOTA_WAIT:
        raise QuotaExhausted(msg[:300])
    return secs + 2


async def with_quota(make_call, label: str, attempts: int = 3):
    """Run make_call() (a coroutine factory), waiting out rate limits; None after other errors."""
    for attempt in range(attempts):
        try:
            return await make_call()
        except QuotaExhausted:
            raise
        except Exception as e:
            wait = quota_wait(e)
            if wait is None:
                print(f"    {label} failed ({attempt + 1}/{attempts}): {str(e)[:150]}")
                continue
            print(f"    rate limit on {label}, waiting {wait:.0f}s")
            await asyncio.sleep(wait)
    return None


# ---------- answers + judge ----------

def build_metrics():
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    openai = config.is_openai(config.JUDGE_MODEL)
    # bypass_n: Groq rejects n > 1, so ragas sends separate requests instead
    judge = LangchainLLMWrapper(config.get_judge_llm(), bypass_n=not openai)
    emb = LangchainEmbeddingsWrapper(config.get_embeddings())
    # answer relevancy averages `strictness` generated questions; 1 on Groq to save quota
    metrics = [LLMContextRecall(), Faithfulness(), ResponseRelevancy(strictness=3 if openai else 1)]
    for m in metrics:
        m.llm = judge
        if hasattr(m, "embeddings"):
            m.embeddings = emb
        m.init(RunConfig(timeout=180, max_workers=1))
    assert [m.name for m in metrics] == LLM_METRICS
    return metrics


async def run_llm(items, retrieved, names):
    from ragas import SingleTurnSample

    answers = {r["key"]: r["answer"] for r in load_jsonl(ANSWERS)}
    scores = load_scores()
    llm = config.get_llm(reasoning_effort="low", max_retries=2)
    metrics = build_metrics()

    for n, it in enumerate(items, 1):
        for name in names:
            row = retrieved[name][it["id"]]
            key = sample_key(it["question"], row["chunk_ids"])
            missing = [m for m in metrics if m.name not in scores.get(key, {})]
            if not missing:
                continue
            print(f"[{n}/{len(items)}] {it['id']} {name}")
            if key not in answers and any(m.name != "context_recall" for m in missing):
                prompt = ANSWER_PROMPT.format(context="\n\n---\n\n".join(row["contexts"]),
                                              question=it["question"])
                msg = await with_quota(lambda: llm.ainvoke(prompt), "answer")
                if msg is None:
                    continue
                answers[key] = msg.content
                append_jsonl(ANSWERS, {"key": key, "id": it["id"], "config": name,
                                       "answer": msg.content})
            sample = SingleTurnSample(user_input=it["question"], retrieved_contexts=row["contexts"],
                                      response=answers.get(key, ""), reference=it["reference"])
            for m in missing:
                if m.name != "context_recall" and key not in answers:
                    continue
                s = await with_quota(lambda: m.single_turn_ascore(sample), m.name)
                if s is None:   # failed: retried on the next run
                    continue
                s = None if s != s else float(s)   # NaN (e.g. "don't know" has no claims) -> null
                scores.setdefault(key, {})[m.name] = s
                append_jsonl(JUDGE, {"key": key, "judge": config.JUDGE_MODEL, "metric": m.name,
                                     "score": s})
    return answers, scores


# ---------- report ----------

def group(it: dict) -> str:
    return "multi_hop" if it["type"] == "multi_hop" else f"single_{it['content_type']}"


def report(items, retrieved, names, answers, scores):
    rows, unjudged = [], set()
    for it in items:
        for name in names:
            r = retrieved[name][it["id"]]
            key = sample_key(it["question"], r["chunk_ids"])
            if set(LLM_METRICS) - set(scores.get(key, {})):
                unjudged.add(it["id"])
            rows.append({"config": name, "id": it["id"], "group": group(it),
                         **retrieval_scores(r["metas"], it["pages"]),
                         **{m: scores.get(key, {}).get(m) for m in LLM_METRICS},
                         "latency_s": r["latency_s"], "chunk_ids": " ".join(r["chunk_ids"]),
                         "question": it["question"], "answer": answers.get(key)})
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "ragas_per_question.csv", index=False)

    # LLM metrics are compared only on questions judged for every config
    complete = {it["id"] for it in items} - unjudged
    summary = []
    for name in names:
        d = df[df.config == name]
        dj = d[d.id.isin(complete)]
        row = {"config": name, "n": len(d)}
        row.update({m: round(d[m].mean(), 3) for m in RETRIEVAL_METRICS})
        row["n_judged"] = len(dj)
        row.update({m: round(dj[m].mean(), 3) if len(dj) else None for m in LLM_METRICS})
        row["p50_s"] = round(statistics.median(d.latency_s), 3)
        row["p95_s"] = round(d.latency_s.quantile(0.95), 3)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(config.RESULTS_DIR / "ragas_summary.csv", index=False)

    by_group = (df.groupby(["config", "group"])[["hit@5", "mrr", "context_recall"]]
                .mean().round(3).unstack("group"))
    by_group.to_csv(config.RESULTS_DIR / "ragas_by_group.csv")
    print("\n" + summary.to_markdown(index=False))
    print("\nBy question group:\n" + by_group.to_markdown())


def main(limit: int | None, names: list[str], no_llm: bool, rerun_retrieval: bool):
    CACHE.mkdir(exist_ok=True)
    items = json.loads(config.EVAL_SET_PATH.read_text(encoding="utf-8"))[:limit]
    retrieved = retrieve_all(items, names, rerun_retrieval)
    answers, scores = {}, {}
    if not no_llm:
        print(f"answers: {config.LLM_MODEL}, judge: {config.JUDGE_MODEL}")
        try:
            answers, scores = asyncio.run(run_llm(items, retrieved, names))
        except QuotaExhausted as e:
            print(f"\nDaily quota reached, progress is cached; re-run later to resume.\n  {e}")
    if no_llm or not answers:   # still report what earlier runs cached
        answers = {r["key"]: r["answer"] for r in load_jsonl(ANSWERS)}
        scores = load_scores()
    report(items, retrieved, names, answers, scores)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--no-llm", action="store_true", help="retrieval metrics only (no API calls)")
    p.add_argument("--rerun-retrieval", action="store_true",
                   help="ignore cached retrieval (needed after re-indexing)")
    a = p.parse_args()
    main(a.limit, a.configs, a.no_llm, a.rerun_retrieval)
