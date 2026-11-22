"""
evaluate.py - runs a list of test questions against a datasheet and prints a score.

Usage:
    set LLM_API_KEY=...   (Windows: set, Mac/Linux: export)
    set LLM_BASE_URL=https://api.groq.com/openai/v1
    set LLM_MODEL=openai/gpt-oss-20b
    python evaluate.py LM358-D.pdf questions_lm358.json

A question passes if the answer contains ANY of the strings in "expect_any" and NONE
of the strings in "reject". For questions that should not be answerable, expect
"could not find". Always read failed answers yourself: string matching is crude.
"""

import json
import sys
import time

from rag import Index, ask_llm, build_messages, chunk_pages, load_pdf_pages


def main(pdf_path: str, questions_path: str) -> None:
    pages = load_pdf_pages(pdf_path)
    index = Index(chunk_pages(pages), pages)
    questions = json.load(open(questions_path, encoding="utf-8"))

    passed = 0
    for i, q in enumerate(questions, start=1):
        hits = index.search(q["question"], top_k=6)
        sent = index.pages_for(hits, max_pages=3)
        answer = ask_llm(build_messages(q["question"], sent))
        low = answer.lower()
        ok = any(e.lower() in low for e in q["expect_any"]) and not any(
            r.lower() in low for r in q.get("reject", [])
        )
        passed += ok
        print(f"{i:>2}. {'PASS' if ok else 'FAIL'}  {q['question']}")
        print(f"      pages sent: {[p for p, _ in sent]}")
        print(f"      answer: {answer}")
        time.sleep(2)  # stay under free-tier rate limits

    print(f"\nScore: {passed}/{len(questions)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python evaluate.py <datasheet.pdf> <questions.json>")
    main(sys.argv[1], sys.argv[2])
