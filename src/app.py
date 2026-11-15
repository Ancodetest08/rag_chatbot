"""Streamlit web UI for the RAG Chatbot.

Run from the project root:
    streamlit run src/app.py
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from chatbot import (
    CHUNKERS,
    DEFAULT_CHUNKER,
    EMBEDDING_MODEL,
    IDK_MESSAGE,
    LLM_MODEL,
    PAPERS_DIR,
    ROOT_DIR,
    TOP_K,
    build_rag_chain,
    source_label,
)

EVAL_FILE = ROOT_DIR / "eval" / "eval_questions.json"

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

load_dotenv(ROOT_DIR / ".env")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading knowledge base (first run builds the index)...")
def load_chain(chunker: str):
    return build_rag_chain(chunker)


def example_questions(limit: int = 5) -> list[str]:
    """Answerable questions from the eval file, skipping unfinished TODO templates."""
    try:
        items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [
        i["question"]
        for i in items
        if i.get("answerable") and not i["question"].strip().upper().startswith("TODO")
    ][:limit]


def serialize_sources(docs) -> list[dict]:
    """Keep label + text of each distinct retrieved chunk.

    Used for display in the history.
    """
    seen, out = set(), []
    for d in docs:
        key = (
            source_label(d),
            d.page_content,
        )
        if key not in seen:
            seen.add(key)
            out.append(
                {
                    "label": source_label(d),
                    "text": d.page_content,
                }
            )
    return out


def render_sources(sources: list[dict]) -> None:
    labels = list(dict.fromkeys(s["label"] for s in sources))
    st.caption("Sources: " + " · ".join(labels))
    with st.expander(f"View retrieved passages ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}**")
            st.text(s["text"])
            st.divider()


def stream_answer(chain, question: str, holder: dict):
    """Yield answer tokens and remember the retrieved docs in `holder`."""
    for chunk in chain.stream(question):
        if "docs" in chunk:
            holder["docs"] = chunk["docs"]
        if "answer" in chunk:
            yield chunk["answer"]


# --------------------------------------------------------------------------- #
# Pre-flight checks
# --------------------------------------------------------------------------- #
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add your key to the .env file, then restart.")
    st.stop()
if not any(PAPERS_DIR.rglob("*.pdf")):
    st.error(f"No PDF files found in `{PAPERS_DIR.name}`. Add your documents there and restart.")
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.caption("Answers only from your private PDF papers, with citations.")

    chunker = st.selectbox(
        "Chunking strategy",
        CHUNKERS,
        index=CHUNKERS.index(DEFAULT_CHUNKER),
        help="Recursive (default): fixed-size chunks with overlap. Semantic: splits where the topic changes.",
    )
    chain, vectorstore = load_chain(chunker)

    st.markdown(
        f"**Chat model:** `{LLM_MODEL}`  \n"
        f"**Embedding model:** `{EMBEDDING_MODEL}`  \n"
        f"**Vector store:** `FAISS`  \n"
        f"**Indexed chunks:** {vectorstore.index.ntotal}  \n"
        f"**Passages per answer:** {TOP_K}"
    )

    if st.button("Rebuild index", help="Re-embed all PDFs (calls the OpenAI API)."):
        with st.spinner("Rebuilding index..."):
            build_rag_chain(chunker, rebuild=True)
            load_chain.clear()
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    examples = example_questions()
    if examples:
        st.divider()
        st.markdown("**Try an example**")
        for q in examples:
            if st.button(q, key=f"ex_{q}"):
                st.session_state.pending = q

    st.divider()
    st.caption("Each question is answered independently (no conversation memory).")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
st.header("Ask your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about the papers...") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        holder: dict = {}
        answer = st.write_stream(stream_answer(chain, question, holder))
        sources = []
        if IDK_MESSAGE not in answer:
            sources = serialize_sources(holder.get("docs", []))
            if sources:
                render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
