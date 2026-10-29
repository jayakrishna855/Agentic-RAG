"""Phase 7: memory accuracy, within a session and after a restart.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config
from agent.graph import ask, build_graph, get_checkpointer

TEST_DB = config.DATA / "memory_test.sqlite"

# (setup turn, follow-up, keywords: the follow-up answer must contain one of them)
CASES = [
    ("My name is Priya and I'm researching NVIDIA.", "What's my name?", ["priya"]),
    ("Let's focus on Apple.", "What is its stock ticker?", ["aapl"]),
    ("What is 120 times 3?", "Now divide that result by 4.", ["90"]),
    ("I care most about Microsoft's risk factors.", "Which company did I say I care about?",
     ["microsoft", "msft"]),
    ("Remember the number 4821.", "What number did I ask you to remember?", ["4821"]),
    ("I have $5,000 to split equally between NVDA and AAPL.", "How much goes to each?",
     ["2,500", "2500"]),
    ("How many employees did NVIDIA have at the end of fiscal 2026?",
     "Remind me, what employee count did you just give me?", ["42,000", "42000"]),
]


def last_answer(graph, question: str, tid: str) -> str:
    return ask(graph, question, tid, source="memory-eval")[-1].content


def main():
    rows = []
    g1 = build_graph(get_checkpointer(TEST_DB))
    for setup, follow, keywords in CASES:
        same, restart = str(uuid.uuid4()), str(uuid.uuid4())

        last_answer(g1, setup, same)
        a_same = last_answer(g1, follow, same)

        last_answer(g1, setup, restart)
        g2 = build_graph(get_checkpointer(TEST_DB))   # fresh graph + connection = app restart
        a_restart = last_answer(g2, follow, restart)

        ok_same = any(k in a_same.lower() for k in keywords)
        ok_restart = any(k in a_restart.lower() for k in keywords)
        rows.append({"setup": setup, "follow_up": follow, "keywords": keywords,
                     "same_session": ok_same, "after_restart": ok_restart,
                     "answer_same": a_same, "answer_restart": a_restart})
        print(f"{'OK' if ok_same else 'X '} {'OK' if ok_restart else 'X '}  {follow}")

    n = len(rows)
    summary = {"n": n, "same_session_acc": round(sum(r["same_session"] for r in rows) / n, 3),
               "after_restart_acc": round(sum(r["after_restart"] for r in rows) / n, 3)}
    (config.RESULTS_DIR / "memory_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Same-session memory accuracy:  {sum(r['same_session'] for r in rows)}/{n}")
    print(f"After-restart memory accuracy: {sum(r['after_restart'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
