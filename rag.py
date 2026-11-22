"""
rag.py - the core logic of the Datasheet Q&A Assistant.

RAG = Retrieval-Augmented Generation. The idea in 4 steps:
  1. Read the PDF and split it into small chunks of text.
  2. Turn every chunk into an "embedding" (a list of numbers that captures meaning).
  3. When the user asks a question, embed the question too and find the chunks
     whose embeddings are most similar (retrieval).
  4. Send ONLY those chunks plus the question to an LLM, and tell it to answer
     from them (generation). This reduces made-up answers (hallucinations).
"""

import os


# ---------- Step 1: read the PDF and split into chunks ----------

def load_pdf_pages(file) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for every page in the PDF.
    `file` can be a path or a file-like object (e.g. a Streamlit upload)."""
    from pypdf import PdfReader  # imported here so other functions work without it

    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((number, text))
    return pages


def chunk_pages(pages, words_per_chunk: int = 180, overlap: int = 40) -> list[dict]:
    """Split each page into overlapping chunks of roughly `words_per_chunk` words.
    Overlap keeps a sentence that falls on a boundary from being cut in half.
    Each chunk remembers its page number so we can cite it later."""
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


# ---------- Steps 2 and 3: embeddings and search ----------

class Index:
    """Holds all chunks and their embeddings, and can search them."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, free, runs on CPU

    def __init__(self, chunks: list[dict]):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.chunks = chunks
        self.model = SentenceTransformer(self.MODEL_NAME)
        # normalize_embeddings=True makes cosine similarity a simple dot product
        self.vectors = self.model.encode(
            [c["text"] for c in chunks], normalize_embeddings=True
        )

    def search(self, question: str, top_k: int = 4) -> list[dict]:
        """Return the top_k chunks most similar to the question."""
        q = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q                      # similarity of q to every chunk
        best = self.np.argsort(scores)[::-1][:top_k]   # indices of highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


# ---------- Step 4: ask the LLM ----------

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about an electronics datasheet. "
    "Use ONLY the provided excerpts. If the answer is not in the excerpts, say "
    "'I could not find that in the document.' Always mention the page number(s) "
    "you used, like (p. 5). Keep units and values exactly as written."
)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    user = f"Excerpts:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ask_llm(messages, api_key=None, base_url=None, model=None) -> str:
    """Call any OpenAI-compatible chat API (Groq, Google Gemini, OpenRouter, OpenAI...).
    Settings come from arguments or from environment variables:
      LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """
    from openai import OpenAI

    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL")
    if not (api_key and base_url and model):
        raise ValueError("Set the API key, base URL and model name first.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.1
    )
    return response.choices[0].message.content
