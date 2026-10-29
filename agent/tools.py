"""Phase 5: the agent's five tools. The LLM picks tools from these docstrings.
"""
import re
import sys
from functools import cache
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool

import config

Ticker = Literal[tuple(config.COMPANIES)]


@cache
def get_retriever():
    from retrieval.retriever import Retriever
    return Retriever()


@cache
def _tavily():
    from langchain_tavily import TavilySearch
    return TavilySearch(max_results=3)


@cache
def _wiki():
    import wikipedia
    from langchain_community.utilities import WikipediaAPIWrapper
    # Wikimedia rate-limits the library's generic User-Agent (HTTP 429); identify the app instead
    wikipedia.set_user_agent("agentic-rag/0.1 (educational RAG project; python-wikipedia)")
    return WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)


@tool
def search_documents(query: str, company: list[Ticker] | None = None) -> str:
    """Search the internal library of annual reports (Form 10-K filings): Apple (AAPL, fiscal
    2025), Amazon (AMZN, fiscal 2025), Microsoft (MSFT, fiscal 2026) and NVIDIA (NVDA, fiscal
    2026). Use for these companies' reported financials, segments, risk factors, strategy,
    employees, or anything stated in their filings.

    Args:
        query: what to look for, with specific terms (e.g. "Data Center revenue fiscal 2026").
            For line items, include the statement name (e.g. "consolidated balance sheets
            total assets", "statements of operations net income").
        company: optional tickers to restrict the search to, e.g. ["NVDA"]. For comparisons,
            search each company separately so both get results.
    """
    try:
        docs = get_retriever().retrieve(query, mode="hybrid_rerank", company=company)
    except Exception as e:
        return f"Search error: {e}"
    if not docs:
        return "No matching passages found."
    # each chunk starts with its own source line, e.g. "Apple (AAPL) Form 10-K FY2025, page 34"
    return "\n\n---\n\n".join(d.page_content for d in docs)


@tool
def web_search(query: str) -> str:
    """Search the live web. Use for recent news, current events, analyst views, or anything
    after the filings' fiscal years, and for companies not in the 10-K library."""
    try:
        res = _tavily().invoke({"query": query})
    except Exception as e:
        return f"Web search error: {e}"
    results = res.get("results", []) if isinstance(res, dict) else []
    if not results:
        return "No web results found."
    return "\n\n".join(f"{r.get('title')}\n{r.get('url')}\n{r.get('content')}" for r in results)


@tool
def wikipedia(query: str) -> str:
    """Look up general encyclopedic background: company history, founders, people, places,
    and definitions of concepts (e.g. "What is a 10-K?"). Not for recent events or figures."""
    try:
        return _wiki().run(query) or "No Wikipedia article found."
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing stock price and daily change for a ticker symbol, e.g. AAPL,
    MSFT, NVDA, AMZN. Input must be the ticker symbol, not the company name."""
    import yfinance as yf
    symbol = ticker.upper().strip()
    try:
        hist = yf.Ticker(symbol).history(period="5d")
    except Exception as e:
        return f"Price lookup error for {symbol}: {e}"
    if hist.empty:
        return f"No price data found for {symbol}. Check that it is a valid ticker symbol."
    last = hist.iloc[-1]
    out = f"{symbol} last close: ${last['Close']:.2f} on {hist.index[-1].date()}"
    if len(hist) > 1:
        prev = hist.iloc[-2]["Close"]
        out += f" ({(last['Close'] - prev) / prev * 100:+.2f}% vs previous close)"
    return out


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '0.15 * 4230000' or '(120.5 - 98.2) / 98.2 * 100'.
    Use for any arithmetic (growth rates, differences, ratios) instead of computing in your head."""
    import numexpr
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression).replace("$", "").replace("^", "**")
    try:
        value = numexpr.evaluate(expr).item()
    except Exception as e:
        return f"Calculation error: {e}"
    if isinstance(value, float) and abs(value) >= 1:
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:.6g}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)


TOOLS = [search_documents, web_search, wikipedia, get_stock_price, calculator]


if __name__ == "__main__":
    checks = [
        (search_documents, {"query": "Data Center revenue fiscal 2026", "company": ["NVDA"]}),
        (web_search, {"query": "NVIDIA latest earnings news"}),
        (wikipedia, {"query": "Jensen Huang"}),
        (get_stock_price, {"ticker": "aapl"}),
        (calculator, {"expression": "60,922 * 100"}),
        (calculator, {"expression": "import os"}),
    ]
    for t, args in checks:
        out = t.invoke(args)
        print(f"=== {t.name}({args}) -> {len(out)} chars\n{out[:300]}\n")
