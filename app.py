"""Phase 9: Streamlit chat UI for the agent.

Changes vs. the original plan:
- Tool calls appear live while the agent works (agent.graph.stream_turn).
- The retriever and reranker are loaded and warmed up once at startup, so the first
  question doesn't pay the cold start seen in the Phase 8 P95 latency.
- The sidebar lists past conversations from SQLite, so they can be reopened.
- Replayed turns show their tool calls and sources too, not just the text.
- Example questions for a quick demo; chat runs are tagged "chat" in LangSmith.

Run:  streamlit run app.py
"""
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import config
from agent.graph import build_graph, get_checkpointer, stream_turn
from agent.tools import get_retriever

st.set_page_config(page_title="Agentic RAG", page_icon="📊")

EXAMPLES = [
    "How did NVIDIA's Data Center revenue change in fiscal 2026?",
    "Compare Apple's and Microsoft's employee counts from their 10-Ks.",
    "What is AMZN trading at, and what was Amazon's net income in 2025?",
    "Who founded NVIDIA, and what's in the news about it this week?",
]


@st.cache_resource(show_spinner="Loading indexes and models...")
def load_graph():
    get_retriever().warmup()
    checkpointer = get_checkpointer()
    checkpointer.setup()   # create tables now, so listing threads works on a fresh database
    return build_graph(checkpointer)


def recent_threads(graph, limit: int = 12) -> list[tuple[str, str]]:
    """(thread_id, first question) of the most recent conversations."""
    rows = graph.checkpointer.conn.execute(
        "SELECT thread_id FROM checkpoints GROUP BY thread_id "
        "ORDER BY MAX(checkpoint_id) DESC LIMIT ?", (limit,)).fetchall()
    threads = []
    for (tid,) in rows:
        msgs = graph.get_state({"configurable": {"thread_id": tid}}).values.get("messages", [])
        first = next((m.content for m in msgs if isinstance(m, HumanMessage)), "(empty)")
        threads.append((tid, first))
    return threads


def turns(messages: list) -> list[list]:
    """Split a thread's messages into turns, each starting with a HumanMessage."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage) or not out:
            out.append([])
        out[-1].append(m)
    return out


def show_tool_call(tc: dict):
    args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
    st.caption(f"🔧 `{tc['name']}({args})`")


def show_sources(tool_msgs: list[ToolMessage]):
    if not tool_msgs:
        return
    with st.expander(f"Sources / tool outputs ({len(tool_msgs)})"):
        for tm in tool_msgs:
            st.markdown(f"**{tm.name}**")
            if tm.name == "search_documents":
                for passage in tm.content.split("\n\n---\n\n"):
                    source, _, body = passage.partition("\n")
                    st.markdown(f"*{source}*")
                    st.text(body[:800])
            else:
                st.text(tm.content[:1500])


def show_turn(turn: list):
    st.chat_message("user").write(turn[0].content)
    with st.chat_message("assistant"):
        for m in turn[1:]:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    show_tool_call(tc)
        answer = next((m.content for m in reversed(turn) if isinstance(m, AIMessage) and m.content), "")
        st.write(answer)
        show_sources([m for m in turn if isinstance(m, ToolMessage)])


def run_turn(graph, question: str):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        tool_msgs, answer = [], ""
        with st.status("Thinking...", expanded=True) as status:
            for m in stream_turn(graph, question, st.session_state.thread_id, source="chat"):
                if isinstance(m, AIMessage):
                    for tc in m.tool_calls:
                        show_tool_call(tc)
                    if m.content:
                        answer = m.content
                elif isinstance(m, ToolMessage):
                    tool_msgs.append(m)
            status.update(label=f"Done · {len(tool_msgs)} tool call(s)", state="complete",
                          expanded=False)
        st.write(answer)
        show_sources(tool_msgs)


graph = load_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Conversations")
    if st.button("➕ New chat", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Current thread: `{st.session_state.thread_id[:8]}`")
    for tid, title in recent_threads(graph):
        label = ("▶ " if tid == st.session_state.thread_id else "") + title[:40]
        if st.button(label, key=f"t-{tid}", width="stretch"):
            st.session_state.thread_id = tid
            st.rerun()
    st.divider()
    st.markdown("**Tools:** 10-K search · Web · Wikipedia · Stock price · Calculator")
    st.caption(f"10-Ks: {', '.join(config.COMPANIES)} · Model: {config.LLM_MODEL}")

st.title("📊 Multi-Source Agentic RAG")
cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
history = graph.get_state(cfg).values.get("messages", [])
for turn in turns(history):
    show_turn(turn)

prompt = st.chat_input("Ask about Apple, Microsoft, NVIDIA, Amazon, markets, or anything else")
if not history and not prompt:
    st.markdown("Try an example:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex-{i}", width="stretch"):
            prompt = ex
if prompt:
    run_turn(graph, prompt)
    st.rerun()   # refresh the sidebar's conversation list
