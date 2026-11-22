"""
rag.py (version 2) - core logic of the Datasheet Q&A Assistant.

What changed from version 1, and why (each change came from a failed test):
  1. Pages are read in "layout" mode and tables are written with " | " between
     columns, so the column headers (LM258 | LM358 | LM358A, Min | Typ | Max)
     stay lined up with the numbers. Fixes: model took the LM258 max value
     (5.0 mV) for the LM358 typical value (2.0 mV).
  2. Search is "hybrid": meaning-based (embeddings) + exact-word (keyword)
     scores are combined. Fixes: table rows that the embedding search missed.
  3. The model receives whole pages (the top few pages found), not tiny
     fragments, so a table row is always seen together with its header.
  4. The prompt tells the model how to read tables and to name the column.
"""

import math
import os
import re
from collections import Counter


# ---------- Step 1: read the PDF ----------

def _clean_layout_text(text: str) -> str:
    """Turn wide layout text into compact rows with ' | ' between columns."""
    text = re.sub(r"Ty\s+p", "Typ", text)          # PDF splits 'Typ' into 'Ty p'
    text = re.sub(r"/C\d{4}", "", text)            # drop unreadable symbol codes like /C0087
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        line = re.sub(r"\s{3,}", " | ", line.strip())  # 3+ spaces = column break
        lines.append(line)
    return "\n".join(lines)


def load_pdf_pages(file) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for every page that has text."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            raw = page.extract_text() or ""
        text = _clean_layout_text(raw)
        if text.strip():
            pages.append((number, text))
    return pages


def chunk_pages(pages, words_per_chunk: int = 150, overlap: int = 30) -> list[dict]:
    """Split pages into overlapping chunks of about `words_per_chunk` words."""
    if overlap >= words_per_chunk:
        raise ValueError("overlap must be smaller than words_per_chunk")

    chunks = []
    step = words_per_chunk - overlap
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = words[start:start + words_per_chunk]
            if not piece:
                break
            chunks.append({"page": page_number, "text": " ".join(piece)})
            if start + words_per_chunk >= len(words):
                break
    return chunks


# ---------- Step 2: keyword scoring (no extra libraries) ----------

STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "what", "which", "for", "to", "in", "on",
    "and", "or", "does", "do", "how", "at", "it", "its", "this", "that", "with",
    "typical", "maximum", "minimum", "value",
}


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class KeywordIndex:
    """A small BM25-style scorer: rewards chunks containing the rare words of the question."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, question: str) -> list[float]:
        terms = tokenize(question)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                f = doc.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / norm
            out.append(s)
        return out


def rank_positions(scores) -> dict[int, int]:
    """Map chunk index -> rank (0 = best)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {i: r for r, i in enumerate(order)}


# ---------- Step 3: hybrid search ----------

class Index:
    """Holds chunks, embeddings and keyword statistics; can search both."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[dict], pages: list[tuple[int, str]], use_embeddings=True):
        import numpy as np

        self.np = np
        self.chunks = chunks
        self.page_text = dict(pages)
        self.keywords = KeywordIndex([c["text"] for c in chunks])
        self.model = None
        self.vectors = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.MODEL_NAME)
            self.vectors = self.model.encode(
                [c["text"] for c in chunks], normalize_embeddings=True
            )

    def search(self, question: str, top_k: int = 6) -> list[dict]:
        """Return top_k chunks using reciprocal rank fusion of both searches."""
        kw_scores = self.keywords.scores(question)
        kw_rank = rank_positions(kw_scores)
        if self.model is not None:
            q = self.model.encode([question], normalize_embeddings=True)[0]
            em_scores = (self.vectors @ q).tolist()
            em_rank = rank_positions(em_scores)
        else:
            em_scores, em_rank = [0.0] * len(self.chunks), {}

        fused = []
        for i in range(len(self.chunks)):
            score = 1 / (60 + kw_rank[i])
            if em_rank:
                score += 1 / (60 + em_rank[i])
            fused.append((score, i))
        fused.sort(reverse=True)
        return [
            {**self.chunks[i], "score": s, "similarity": em_scores[i], "keyword": kw_scores[i]}
            for s, i in fused[:top_k]
        ]

    def pages_for(self, hits: list[dict], max_pages: int = 3) -> list[tuple[int, str]]:
        """Whole pages of the best hits (in order of relevance), so tables keep their headers."""
        seen = []
        for h in hits:
            if h["page"] not in seen:
                seen.append(h["page"])
            if len(seen) == max_pages:
                break
        return [(p, self.page_text[p]) for p in seen]


# ---------- Step 4: ask the LLM ----------

SYSTEM_PROMPT = (
    "You answer questions about an electronics datasheet using ONLY the pages provided. "
    "If the answer is not in them, say 'I could not find that in the document.' "
    "Never use numbers from graphs' axis labels.\n"
    "How to read the tables: columns are separated by ' | ' and go left to right. "
    "A datasheet often has several device variants side by side (for example LM258, LM358, "
    "LM358A), and each variant has its own Min, Typ and Max columns, in that order. Count "
    "the columns carefully, use the header row to find the right variant and the right "
    "Min/Typ/Max column, and never mix variants or take a Max value when Typ was asked. "
    "'-' means no value is given.\n"
    "Answer in one or two short sentences. State the value with its unit, the variant "
    "and the condition if relevant, and the page number like (p. 4)."
)


def build_messages(question: str, pages: list[tuple[int, str]]) -> list[dict]:
    context = "\n\n".join(f"=== PAGE {n} ===\n{t}" for n, t in pages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Datasheet pages:\n{context}\n\nQuestion: {question}"},
    ]


def ask_llm(messages, api_key=None, base_url=None, model=None) -> str:
    """Call any OpenAI-compatible chat API. Settings may also come from the
    environment variables LLM_API_KEY, LLM_BASE_URL and LLM_MODEL."""
    from openai import OpenAI

    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL")
    if not (api_key and base_url and model):
        raise ValueError("Set the API key, base URL and model name first.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    answer = (response.choices[0].message.content or "").strip()
    if not answer:  # some reasoning models sometimes return an empty message
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0
        )
        answer = (response.choices[0].message.content or "").strip()
    return answer or "(The model returned an empty answer. Please ask again.)"
