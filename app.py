"""
app.py (version 2) - web interface. Run with:  streamlit run app.py
"""

import io

import streamlit as st

from rag import Index, ask_llm, build_messages, chunk_pages, load_pdf_pages

st.set_page_config(page_title="Datasheet Q&A Assistant", page_icon="🔌")
st.title("🔌 Datasheet Q&A Assistant")
st.caption("Upload an electronics datasheet (PDF) and ask questions about it.")

with st.sidebar:
    st.header("LLM settings")
    st.write("Any OpenAI-compatible API works.")
    base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1")
    model = st.text_input("Model name", value="openai/gpt-oss-20b")
    api_key = st.text_input("API key", type="password")
    top_k = st.slider("Chunks searched", 4, 12, 6)
    max_pages = st.slider("Pages sent to the LLM", 1, 5, 3)


@st.cache_resource(show_spinner="Reading and indexing the PDF...")
def build_index(file_bytes: bytes, name: str) -> Index:
    pages = load_pdf_pages(io.BytesIO(file_bytes))
    if not pages:
        raise ValueError("No text found. Scanned PDFs need OCR, which is not supported.")
    # Use meaning-based search too when sentence-transformers is installed.
    # On small free hosts it is left out and keyword search alone is used.
    try:
        import sentence_transformers  # noqa: F401
        use_embeddings = True
    except ImportError:
        use_embeddings = False
    return Index(chunk_pages(pages), pages, use_embeddings=use_embeddings)


uploaded = st.file_uploader("Datasheet PDF", type="pdf")

if uploaded:
    try:
        index = build_index(uploaded.getvalue(), uploaded.name)
    except Exception as err:
        st.error(str(err))
        st.stop()

    mode = "keyword + meaning search" if index.model is not None else "keyword search only"
    st.success(f"Indexed {len(index.chunks)} text chunks from {uploaded.name} ({mode})")
    question = st.text_input("Ask a question", placeholder="What is the maximum supply voltage?")

    if question:
        hits = index.search(question, top_k=top_k)
        pages_sent = index.pages_for(hits, max_pages=max_pages)
        try:
            with st.spinner("Asking the LLM..."):
                answer = ask_llm(build_messages(question, pages_sent), api_key, base_url, model)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as err:
            st.error(f"LLM call failed: {err}")

        with st.expander("Pages sent to the LLM"):
            st.write("Pages: " + ", ".join(str(p) for p, _ in pages_sent))
            for p, text in pages_sent:
                st.markdown(f"**Page {p}**")
                st.code(text, language=None)
else:
    st.info("Try the sample in sample/, or download a datasheet (LM358, ATmega328P, ESP32) from the manufacturer's site.")
