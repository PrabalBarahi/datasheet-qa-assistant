# 🔌 Datasheet Q&A Assistant

Upload an electronics datasheet (PDF), ask questions in plain English, and get answers from an LLM that is told to use **only the document**, with page numbers cited.

Built with Python, Streamlit, Sentence-Transformers and any OpenAI-compatible LLM API (tested with Groq, `openai/gpt-oss-20b`).

## How it works
```
PDF -> read each page in layout mode (keeps table columns) -> overlapping chunks
Question -> hybrid search (embeddings + BM25-style keywords, fused by rank)
         -> best pages sent whole to the LLM with a table-reading prompt
         -> short answer with page number
```
| File | Purpose |
|------|---------|
| `rag.py` | PDF reading, chunking, hybrid search, prompt, LLM call |
| `app.py` | Streamlit web interface |
| `evaluate.py` + `questions_lm358.json` | Automatic test runner and 16 test questions |
| `sample/` | A short fictional datasheet for a quick demo |

## Run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt   # full local version; requirements.txt alone gives keyword-only search (used for free hosting)
streamlit run app.py
```
Enter your API key in the sidebar (never commit keys).

## Evaluation
```bash
export LLM_API_KEY=...  LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=openai/gpt-oss-20b
python evaluate.py LM358-D.pdf questions_lm358.json
```
Tested on the onsemi LM358 datasheet (16 pages). Questions include table lookups,
package and pin facts, and questions the document cannot answer (price, slew rate).

| Version | Score | Notes |
|---------|-------|-------|
| v1 (word chunks, 4 excerpts) | 8/11 correct (manual) | Mixed up table columns; missed a table row; read a graph axis as a value |
| v2 (this version) | not yet confirmed | Layout-aware tables, hybrid search, whole pages, column-aware prompt |

## What I learned
- Flattened PDF tables lose column structure, so the model read the LM258 *max* value for the LM358 *typical* one.
- Pure embedding search can miss exact-term table rows; adding keyword search helps.
- Small chunks separate table rows from their headers; sending whole pages fixes that.
- When the answer is missing the model must say so; graphs and figures are a source of invented numbers.

## Limitations
Scanned PDFs need OCR; graphs are not understood; free-tier rate limits apply.

## Author
Prabal Barahi, B.E. Electrical Engineering, IOE Pulchowk Campus
