"""
app.py - the web interface (built with Streamlit).
Run locally with:  streamlit run app.py
"""

import streamlit as st

from rag import Index, ask_llm, build_messages, chunk_pages, load_pdf_pages

st.set_page_config(page_title="Datasheet Q&A Assistant", page_icon="🔌")
st.title("🔌 Datasheet Q&A Assistant")
st.caption("Upload an electronics datasheet (PDF) and ask questions about it.")

# ----- Sidebar: LLM settings (kept here so no secret key is stored in the code) -----
with st.sidebar:
    st.header("LLM settings")
    st.write("Any OpenAI-compatible API works. Example presets are in the README.")
    base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1")
    model = st.text_input("Model name", value="", placeholder="copy a model name from your provider")
    api_key = st.text_input("API key", type="password")
    top_k = st.slider("Excerpts sent to the LLM", 2, 8, 4)


# ----- Build the search index once per uploaded file -----
@st.cache_resource(show_spinner="Reading and indexing the PDF...")
def build_index(file_bytes: bytes, name: str) -> Index:
    import io

    pages = load_pdf_pages(io.BytesIO(file_bytes))
    if not pages:
        raise ValueError("No text found. Scanned PDFs need OCR, which is not supported.")
    return Index(chunk_pages(pages))


uploaded = st.file_uploader("Datasheet PDF", type="pdf")

if uploaded:
    try:
        index = build_index(uploaded.getvalue(), uploaded.name)
    except Exception as err:
        st.error(str(err))
        st.stop()

    st.success(f"Indexed {len(index.chunks)} text chunks from {uploaded.name}")
    question = st.text_input("Ask a question", placeholder="What is the maximum supply voltage?")

    if question:
        hits = index.search(question, top_k=top_k)
        try:
            with st.spinner("Asking the LLM..."):
                answer = ask_llm(build_messages(question, hits), api_key, base_url, model)
            st.markdown("### Answer")
            st.write(answer)
        except Exception as err:
            st.error(f"LLM call failed: {err}")

        with st.expander("Sources used (retrieved excerpts)"):
            for h in hits:
                st.markdown(f"**Page {h['page']}** (similarity {h['score']:.2f})")
                st.write(h["text"])
else:
    st.info("Try a free datasheet, e.g. for the LM358, ATmega328P or ESP32, from the manufacturer's website.")
