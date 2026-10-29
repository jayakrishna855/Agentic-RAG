"""Phase 2: one Retriever with three modes: dense, hybrid (dense + BM25 via RRF), hybrid_rerank.

Changes vs. the original plan:
- BM25 uses rank_bm25 directly with a lowercasing tokenizer that keeps figures like
  "35,934" and "2.5" as single tokens (LangChain's default only splits on whitespace).
- Optional company filter (ticker or list of tickers), applied to both dense and BM25.
- warmup() loads the embedding and reranker models so timings exclude model loading.
- The reranker model is a constructor argument (and the `rerank_model` attribute can be
  switched later; loaded models are cached) so Phase 4 can compare a large and a fast one.
"""
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config

MODES = ("dense", "hybrid", "hybrid_rerank")
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def bm25_tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def as_list(company: str | list[str] | None) -> list[str] | None:
    if company is None:
        return None
    tickers = [company] if isinstance(company, str) else list(company)
    tickers = [t.upper().strip() for t in tickers]
    unknown = [t for t in tickers if t not in config.COMPANIES]
    if unknown:
        raise ValueError(f"Unknown ticker(s) {unknown}; choose from {list(config.COMPANIES)}")
    return tickers


class Retriever:
    def __init__(self, rerank_model: str = config.RERANK_MODEL):
        self.rerank_model = rerank_model
        self.vs = Chroma(
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=config.get_embeddings(),
            collection_name=config.COLLECTION,
        )
        with open(config.CHUNKS_PATH, "rb") as f:
            self.chunks: list[Document] = pickle.load(f)
        self.bm25 = BM25Okapi([bm25_tokenize(c.page_content) for c in self.chunks])
        self.companies = np.array([c.metadata["company"] for c in self.chunks])
        self._rerankers = {}

    @property
    def reranker(self):
        if self.rerank_model not in self._rerankers:
            from sentence_transformers import CrossEncoder
            self._rerankers[self.rerank_model] = CrossEncoder(self.rerank_model, max_length=512)
        return self._rerankers[self.rerank_model]

    def warmup(self):
        self.retrieve("warm up", mode="hybrid_rerank")

    def _dense(self, query: str, k: int, companies: list[str] | None) -> list[Document]:
        where = None
        if companies:
            where = ({"company": companies[0]} if len(companies) == 1
                     else {"company": {"$in": companies}})
        return self.vs.similarity_search(query, k=k, filter=where)

    def _sparse(self, query: str, k: int, companies: list[str] | None) -> list[Document]:
        scores = self.bm25.get_scores(bm25_tokenize(query))
        if companies:
            scores = np.where(np.isin(self.companies, companies), scores, -np.inf)
        top = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top if scores[i] > 0]

    @staticmethod
    def _rrf(result_lists: list[list[Document]], k: int = config.RRF_K) -> list[Document]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d))."""
        scores, docs = {}, {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                cid = doc.metadata["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                docs[cid] = doc
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [docs[cid] for cid in ranked]

    def _rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]

    def retrieve(self, query: str, mode: str = "hybrid_rerank", top_k: int = config.FINAL_K,
                 company: str | list[str] | None = None) -> list[Document]:
        companies = as_list(company)
        if mode == "dense":
            return self._dense(query, top_k, companies)
        fused = self._rrf([self._dense(query, config.CANDIDATES_K, companies),
                           self._sparse(query, config.CANDIDATES_K, companies)])
        if mode == "hybrid":
            return fused[:top_k]
        if mode == "hybrid_rerank":
            return self._rerank(query, fused[: config.CANDIDATES_K], top_k)
        raise ValueError(f"Unknown mode {mode}; choose from {MODES}")

    def timed_retrieve(self, query: str, mode: str, **kwargs):
        t0 = time.perf_counter()
        docs = self.retrieve(query, mode, **kwargs)
        return docs, time.perf_counter() - t0


if __name__ == "__main__":
    r = Retriever()
    r.warmup()
    queries = [
        ("What are the main risk factors related to supply chain?", None),
        ("Apple total net sales iPhone 2025", None),
        ("NVIDIA Data Center revenue fiscal 2026", None),
        ("How many full-time employees does the company have?", "AMZN"),
    ]
    for q, company in queries:
        print(f"\n##### {q}" + (f"  [company={company}]" if company else ""))
        for m in MODES:
            docs, secs = r.timed_retrieve(q, m, company=company)
            print(f"=== {m} ({secs * 1000:.0f} ms) ===")
            for d in docs:
                body = d.page_content.splitlines()[1:]
                print(f"  - [{d.metadata['chunk_id']} {d.metadata['type']}] {' '.join(body)[:110]}")
