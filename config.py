import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
CHROMA_DIR = DATA / "chroma"
MD_CACHE_DIR = DATA / "md_cache"     # per-PDF markdown, so re-chunking skips PDF parsing
CHUNKS_PATH = DATA / "chunks.pkl"
EVAL_SET_PATH = DATA / "eval_set.json"
ROUTING_SET_PATH = DATA / "routing_set.json"
MEMORY_DB = DATA / "memory.sqlite"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Agent / answer model. OpenAI by default since Groq's free tier caps gpt-oss-120b at
# 200K tokens/day (agent turns with tool results hit it quickly). Phase 4 answers used
# openai/gpt-oss-120b; set LLM_MODEL=openai/gpt-oss-120b to go back to Groq.
LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-4.1-mini"
# RAGAS judge: a different model family than the answer model (no self-grading bias).
# OpenAI by default (needs OPENAI_API_KEY) so judging doesn't use the Groq free-tier quota;
# set JUDGE_MODEL=qwen/qwen3.8-27b to judge on Groq instead.
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or "gpt-4.1-mini"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"                    # ~6.8 s / query on this CPU
RERANK_MODEL_FAST = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~1.2 s / query; compared in Phase 4

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TABLE_MAX_CHARS = 2400  # larger tables are split by rows, repeating the header in each piece
CANDIDATES_K = 20      # retrieved before fusion/rerank
FINAL_K = 5            # chunks passed to the LLM
RRF_K = 60
COLLECTION = "filings"

# Ticker (PDF filename prefix, e.g. AAPL_10K_2025.pdf) -> company name.
# Used for chunk headers, metadata filters, and the system prompt.
COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon"
}


def is_openai(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4"))


def is_reasoning(model: str) -> bool:
    return model.startswith(("openai/gpt-oss", "qwen/", "gpt-5", "o1", "o3", "o4"))


def get_llm(model: str = LLM_MODEL, temperature: float = 0, reasoning_effort: str | None = None,
            max_retries: int = 3):
    """OpenAI models via langchain-openai (needs OPENAI_API_KEY), anything else via Groq.
    reasoning_effort ("low"/"medium"/"high") applies to reasoning models like gpt-oss and is
    ignored for others; "low" cuts tokens for bulk jobs. Qwen 3 also accepts "none"."""
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and is_reasoning(model) else {}
    if is_openai(model):
        from langchain_openai import ChatOpenAI
        if is_reasoning(model):   # OpenAI reasoning models reject a custom temperature
            return ChatOpenAI(model=model, max_retries=max_retries, **extra)
        return ChatOpenAI(model=model, temperature=temperature, max_retries=max_retries)
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=temperature, max_retries=max_retries, **extra)


def get_judge_llm(model: str = JUDGE_MODEL):
    """Judge LLM for RAGAS: minimal reasoning on Groq models to save tokens."""
    if is_openai(model):
        return get_llm(model, max_retries=6)
    return get_llm(model, reasoning_effort="none" if "qwen" in model else "low", max_retries=2)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
