"""Phase 6: LangGraph agent (agent node <-> tool node loop) with SQLite memory per thread_id.
"""
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, trim_messages)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from agent.prompts import SYSTEM_PROMPT
from agent.tools import TOOLS

RECURSION_LIMIT = 12      # graph steps per turn (each tool round trip is 2 steps)
HISTORY_TOKENS = 2000     # budget for earlier turns in each LLM call
RUN_NAME = "research_agent"   # root run name in LangSmith (default would be "LangGraph")
STEP_LIMIT_MSG = ("Sorry, I couldn't finish that within the step limit. "
                  "Try asking a narrower question.")


def get_checkpointer(path=config.MEMORY_DB):
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def llm_view(messages: list) -> list:
    """Current turn in full; earlier turns as question/answer pairs within HISTORY_TOKENS."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    past = [m for m in messages[:last_human]
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]
    past = trim_messages(past, max_tokens=HISTORY_TOKENS, strategy="last", start_on="human",
                         token_counter=count_tokens_approximately)
    return past + messages[last_human:]


def build_graph(checkpointer=None):
    llm = config.get_llm().bind_tools(TOOLS)

    def agent(state: MessagesState):
        system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat()))
        return {"messages": [llm.invoke([system] + llm_view(state["messages"]))]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)   # routes to "tools" or END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def stream_turn(graph, question: str, thread_id: str = "default", source: str = "chat"):
    """Run one turn, yielding its messages as they are produced: the question, each AI
    message (tool calls / answer) and each tool result. A partial turn survives a
    recursion error. `source` tags the LangSmith trace (chat / routing-eval / memory-eval)."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT,
           "run_name": RUN_NAME, "tags": [source],
           "metadata": {"source": source, "model": config.LLM_MODEL, "thread_id": thread_id}}
    human = HumanMessage(question)
    yield human
    try:
        for update in graph.stream({"messages": [human]}, cfg, stream_mode="updates"):
            for node_output in update.values():
                yield from (node_output or {}).get("messages", [])
    except GraphRecursionError:
        yield AIMessage(STEP_LIMIT_MSG)


def ask(graph, question: str, thread_id: str = "default", source: str = "chat") -> list:
    """Run one turn; returns this turn's messages (question, tool calls/results, answer)."""
    return list(stream_turn(graph, question, thread_id, source))


def tool_calls(turn: list) -> list[dict]:
    return [tc for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls]


if __name__ == "__main__":
    graph = build_graph(get_checkpointer())
    tid = input("thread id (blank = default): ").strip() or "default"
    print("Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        turn = ask(graph, q, tid)
        for tc in tool_calls(turn):
            print(f"  [tool] {tc['name']}({tc['args']})")
        print("Bot:", turn[-1].content)
