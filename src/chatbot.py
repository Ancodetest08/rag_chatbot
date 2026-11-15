"""RAG chatbot over a private PDF knowledge base.

Pipeline: load PDFs -> chunk -> embed -> FAISS index (cached on disk) -> retrieve top-k chunks -> prompt LLM with context -> answer + sources.

Usage:
    python src/chatbot.py                       # recursive chunking (default)
    python src/chatbot.py --chunker semantic    # semantic chunking
    python src/chatbot.py --rebuild             # force re-building the index
    python src/chatbot.py --debug               # show retrieved chunks + distances
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT_DIR / "papers"
INDEX_DIR = ROOT_DIR / "faiss_index"

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

TOP_K = 5
CHUNKERS = ("recursive", "semantic")
DEFAULT_CHUNKER = CHUNKERS[0]

IDK_MESSAGE = "I don't know based on the provided documents."
PROMPT_TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    f"2) If the answer is not clearly contained in the context, reply exactly: '{IDK_MESSAGE}'\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) Cite the sources you used as (file name, page N), using the labels that appear in square brackets before each context passage.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def load_documents(papers_dir: Path = PAPERS_DIR):
    """Load every PDF under `papers_dir` (one Document per page)."""
    loader = DirectoryLoader(
        path=str(papers_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def papers_fingerprint(papers_dir: Path = PAPERS_DIR) -> str:
    """Hash of file names + sizes + mtimes, used to detect a stale index."""
    entries = [
        (str(p.relative_to(papers_dir)), p.stat().st_size, int(p.stat().st_mtime))
        for p in sorted(papers_dir.rglob("*.pdf"))
    ]
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def get_splitter(chunker: str, embeddings: OpenAIEmbeddings):
    if chunker == "semantic":
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
    if chunker == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
        )
    raise ValueError(f"Unknown chunker '{chunker}'. Choose from {CHUNKERS}.")


def build_or_load_vectorstore(embeddings: OpenAIEmbeddings, chunker: str = DEFAULT_CHUNKER, rebuild: bool = False) -> FAISS:
    """Load the cached FAISS index, or build it (and cache it) when missing/stale.

    The index is stored per chunker in faiss_index/<chunker>/ so the two strategies can be compared without overwriting each other.

    FAISS uses L2 distance by default. OpenAI embeddings are unit-length, so ranking by L2 is equivalent to ranking by cosine similarity.
    """
    index_dir = INDEX_DIR / chunker
    manifest = index_dir / "manifest.json"

    fingerprint = papers_fingerprint()

    if not rebuild and (index_dir / "index.faiss").exists() and manifest.exists():
        saved = json.loads(manifest.read_text())
        if (saved.get("fingerprint") == fingerprint and saved.get("embedding_model") == EMBEDDING_MODEL):
            print(f"Loading cached index from faiss_index/{chunker}/")
            # Safe here: we only load an index that this script wrote itself.
            return FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        print("Papers or embedding model changed -> rebuilding index.")

    print(f"Building index with '{chunker}' chunking...")
    docs = load_documents()
    splits = get_splitter(chunker, embeddings).split_documents(docs)
    print(f"\nLoaded {len(docs)} pages -> {len(splits)} chunks.")

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    vectorstore.save_local(str(index_dir))
    manifest.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "embedding_model": EMBEDDING_MODEL,
                "chunker": chunker,
                "num_chunks": len(splits),
            },
            indent=2,
        )
    )
    return vectorstore


# --------------------------------------------------------------------------- #
# Chain
# --------------------------------------------------------------------------- #
def source_label(doc) -> str:
    """Human-readable label such as 'paper_03.pdf, page 4'.

    PyPDFLoader pages start at 0.
    """
    name = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{name}, page {page + 1}" if isinstance(page, int) else name


def format_docs(docs) -> str:
    """Turn retrieved Documents into a clean, citable context block."""
    return "\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


def build_chain(vectorstore: FAISS):
    """Return a chain mapping a question -> {"question", "docs", "answer"}."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    answer_chain = (
        RunnableLambda(
            lambda x: {
                "context": format_docs(x["docs"]),
                "question": x["question"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        docs=retriever,
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer_chain)


def build_rag_chain(chunker: str = DEFAULT_CHUNKER, rebuild: bool = False):
    """Convenience wrapper used by the CLI and by evaluate.py."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = build_or_load_vectorstore(embeddings, chunker, rebuild)
    return build_chain(vectorstore), vectorstore


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def check_environment() -> None:
    load_dotenv(ROOT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found. Add your key to the .env file.")
    if not any(PAPERS_DIR.rglob("*.pdf")):
        sys.exit(f"No PDF files found in {PAPERS_DIR}. Add your documents there first.")


def print_answer(result: dict, debug: bool, vectorstore: FAISS) -> None:
    print(f"\n{result['answer']}\n")

    if IDK_MESSAGE not in result["answer"]:
        sources = list(dict.fromkeys(source_label(d) for d in result["docs"]))
        print("Sources: " + "; ".join(sources) + "\n")
        
    if debug:
        print("--- retrieved chunks (lower distance = closer) ---")
        for doc, dist in vectorstore.similarity_search_with_score(result["question"], k=TOP_K):
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"    {dist:.3f} | {source_label(doc)} | {preview}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Chatbot over PDF papers")
    parser.add_argument(
        "--chunker",
        choices=CHUNKERS,
        default=DEFAULT_CHUNKER,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force re-building the index",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print retrieved chunks",
    )
    args = parser.parse_args()

    check_environment()
    chain, vectorstore = build_rag_chain(args.chunker, args.rebuild)

    print("\nRAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        result = chain.invoke(question)
        print_answer(result, args.debug, vectorstore)


if __name__ == "__main__":
    main()
