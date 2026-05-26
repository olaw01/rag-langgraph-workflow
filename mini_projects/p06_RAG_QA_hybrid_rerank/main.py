import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


# -----------------------
# Loading + chunking
# -----------------------

def load_documents(docs_dir: Path):
    """Load .md files from a folder into a list of Documents."""
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def chunk_documents(docs):
    """Split documents into chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, persist_dir: Path):
    """Create a local Chroma vector store on disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="p06_docs",
    )


def format_context(docs) -> Tuple[str, List[str]]:
    """Combine docs into a context string and collect unique sources."""
    sources = []
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        sources.append(src)
        parts.append(f"[source={src}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts), sorted(set(sources))


# -----------------------
# Hybrid candidates: BM25 + Vector
# -----------------------

def hybrid_candidates(question: str, bm25: BM25Retriever, vec_retriever, fetch_k: int = 25):
    """
    Return a merged, de-duplicated candidate list from:
    - BM25 (keyword/sparse) for exact phrases/acronyms/IDs
    - Vector (dense) for semantic similarity
    """
    # BM25 candidates (keyword)
    bm25.k = fetch_k
    bm25_docs = bm25.invoke(question)

    # Vector candidates (semantic)
    vec_docs = vec_retriever.invoke(question)

    # De-duplicate using a simple heuristic (source + prefix)
    seen = set()
    merged = []
    for d in bm25_docs + vec_docs:
        key = (d.metadata.get("source", "unknown"), d.page_content[:200])
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged


# -----------------------
# LLM reranking (no Cohere needed)
# -----------------------

class RerankPick(BaseModel):
    indices: List[int] = Field(description="Selected indices (can be empty).")
    reason: str = Field(description="Short reason for the selection.")


def llm_rerank(question: str, docs, top_n: int = 5):
    """
    Ask the LLM to pick the best passages among candidates.
    If none contains the answer, it should return an empty list of indices.
    """
    items = []
    for i, d in enumerate(docs):
        snippet = d.page_content[:500].replace("\n", " ")
        src = d.metadata.get("source", "unknown")
        items.append(f"{i}) source={src} text={snippet}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict reranker.\n"
                "Select passages that contain information needed to answer the question.\n"
                "If NONE of the candidates contains the answer, return an EMPTY indices list.\n"
                f"Return up to {top_n} indices.",
            ),
            ("human", "QUESTION:\n{question}\n\nCANDIDATES:\n{candidates}\n\nReturn indices."),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(RerankPick)
    chain = prompt | model
    out: RerankPick = chain.invoke({"question": question, "candidates": "\n".join(items)})

    chosen = [docs[i] for i in out.indices if 0 <= i < len(docs)]
    return chosen, out.reason


# -----------------------
# Final grounded answer
# -----------------------

def rag_answer(question: str, docs) -> dict:
    context, sources = format_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Q&A assistant.\n"
                "RULES:\n"
                "1) Answer ONLY using the provided CONTEXT.\n"
                "2) If the CONTEXT does not contain the answer, say exactly: "
                "\"I don't know based on the provided documents.\"\n"
                "3) Do not use general knowledge. Do not guess. Do not add facts not present in the CONTEXT.\n"
                "4) Keep the answer short and direct.\n"
                "5) Add a final line: EVIDENCE: <one sentence quoted or closely paraphrased from the CONTEXT>.\n",
            ),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    return {"answer": answer, "sources": sources}


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    base_dir = Path(__file__).parent
    docs_dir = base_dir / "data" / "docs"
    persist_dir = base_dir / "data" / "chroma"

    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)

    vectorstore = build_vectorstore(chunks, persist_dir)
    vec_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    bm25 = BM25Retriever.from_documents(chunks)

    print("Hybrid + LLM rerank RAG is ready. Type a question (or 'exit').")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        candidates = hybrid_candidates(q, bm25=bm25, vec_retriever=vec_retriever, fetch_k=25)
        top_docs, reason = llm_rerank(q, candidates, top_n=5)

        # If reranker found nothing, do not hallucinate
        if not top_docs:
            print("\nAnswer:\nI don't know based on the provided documents.\nEVIDENCE: (no supporting passage found)")
            print("\nSources:\n(none)")
            continue

        result = rag_answer(q, top_docs)
        print("\n[rerank reason]", reason)
        print("\nAnswer:\n", result["answer"])
        print("\nSources:\n", "\n".join(result["sources"]))


if __name__ == "__main__":
    main()