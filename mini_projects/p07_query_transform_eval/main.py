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


#---------------------------------------------------------------------------------------------------------------------
# Load + chunk + store
# section: load docs, chunk them, build the index
#---------------------------------------------------------------------------------------------------------------------


# loads documents from directory docs_dir
def load_documents(docs_dir: Path):

    # create a loader that scans a directory
    loader = DirectoryLoader(
        str(docs_dir), # convert Path to a string path
        glob="**/*.md",
        loader_cls=TextLoader, # each file is read using TextLoader
        loader_kwargs={"encoding": "utf-8"}, # UTF-8 encoding to avoid character issues
        show_progress=True, # shows progress while loading
    )
    return loader.load() # loads files and returns a list of Document objects


# splits documents into smaller chunks
def chunk_documents(docs):

    # create a splitter that splits text using hierarchical separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # target chunk size (characters here; conceptually you want manageable chunks)
        chunk_overlap=120, # overlap between chunks to preserve context at boundaries
        separators=["\n\n", "\n", ". ", " ", ""], # split priority: paragraphs, then lines, then sentences, etc.
    )
    return splitter.split_documents(docs) # returns a list of chunk Documents

# builds a Chroma vector store and persists it to persist_dir
def build_vectorstore(chunks, persist_dir: Path):

    # create OpenAI embeddings model (text -> vectors)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # build a vector DB from chunk documents
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="p07_docs",
    )

# formats docs into one context string + returns list of sources
def format_context(docs) -> Tuple[str, List[str]]:
    sources = [] # sources list (e.g file paths)
    parts = []
    for d in docs: # iterate over docs (chunks)
        src = d.metadata.get("source", "unknown") # read source from metadata (set by loader)
        sources.append(src)
        parts.append(f"[source={src}]\n{d.page_content}") # add chunk text with a source header
    return "\n\n---\n\n".join(parts), sorted(set(sources)) # join context with separators and return unique sources.


#---------------------------------------------------------------------------------------------------------------------
# Query transformation (Multi-Query)
# Section: generate multiple query to improve retrieval
#---------------------------------------------------------------------------------------------------------------------

# Pydantic schema: model must return alternative queries list
class MultiQuery(BaseModel):
    queries: List[str] = Field(description="Alternative query rewrites", min_length=3, max_length=5) # queries must have 3–5 strings (format control)

# takes the user question and generates 3–5 search rewrites
def generate_multi_queries(question: str) -> List[str]:

    # build a prompt for the query rewriter
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Generate 3 to 5 alternative search queries that would help retrieve relevant passages.\n"
             "Keep them short. Keep meaning. Include synonyms.\n"
             "Return JSON according to schema."),
            ("human", "User question:\n{question}"),
        ]
    )

    # model returns structured output per MultiQuery
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(MultiQuery)

    # LCEL: prompt -> model
    chain = prompt | model

    # invoke rewriter: get a MultiQuery object with out.queries list
    out: MultiQuery = chain.invoke({"question": question})

    # include the original question as well
    return [question] + [q.strip() for q in out.queries if q.strip()]


#---------------------------------------------------------------------------------------------------------------------
# Retrieval (Hybrid BM25 + Vector) no duplicates
#---------------------------------------------------------------------------------------------------------------------

# hybrid retrieval for a single query
def hybrid_candidates(query: str, bm25: BM25Retriever, vec_retriever, fetch_k: int = 20):
    # set how many results BM25 should return.
    bm25.k = fetch_k
    # get BM25 (keyword) documents
    bm25_docs = bm25.invoke(query)
    # get vector (semantic) documents
    vec_docs = vec_retriever.invoke(query)

    seen = set() # set for deduplication
    merged = [] # merged candidate list
    for d in bm25_docs + vec_docs: # iterate through combined BM25 + vector docs.
        key = (d.metadata.get("source", "unknown"), d.page_content[:200]) # build a dedupe key: source + text prefix
        if key not in seen:
            seen.add(key)
            merged.append(d)
    return merged


#---------------------------------------------------------------------------------------------------------------------
# Simple LLM rerank (OpenAI only)
# Section: LLM-based reranking (no Cohere)
#---------------------------------------------------------------------------------------------------------------------

# reranker output schema: index + reason
class RerankPick(BaseModel):
    indices: List[int] = Field(description="Selected indices (can be empty).")
    reason: str = Field(description="Short reason.")

# select top docs for the question
def llm_rerank(question: str, docs, top_n: int = 5):

    items = [] # bild candidate list for the model to score

    for i, d in enumerate(docs): # enumerate candidates with indices
        snippet = d.page_content[:500].replace("\n", " ") # trim to 500 chars and flatten newlines to keep prompt small
        src = d.metadata.get("source", "unknown") # get source
        items.append(f"{i}) source={src} text={snippet}") # append a candidate list

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a strict reranker.\n"
             "Pick passages that contain information needed to answer the question.\n"
             "If NONE contains the answer, return an EMPTY indices list.\n"
             f"Return up to {top_n} indices."),
            ("human", "QUESTION:\n{question}\n\nCANDIDATES:\n{candidates}"),
        ]
    )

    # reranker model returns RerankPick structured output
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(RerankPick)

    # ivoke reranker; candidates are joined into one newline-separated block
    out: RerankPick = (prompt | model).invoke({"question": question, "candidates": "\n".join(items)})

    # map indices back to documents with range safety
    chosen = [docs[i] for i in out.indices if 0 <= i < len(docs)]

    # return chosen docs + reason
    return chosen, out.reason


#---------------------------------------------------------------------------------------------------------------------
# Answer (strict grounding)
# Section: answer generation with strict grounding
#---------------------------------------------------------------------------------------------------------------------

# Generates an answer using only these docs
def grounded_answer(question: str, docs) -> dict:

    context, sources = format_context(docs) # Build context and sources

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a Q&A assistant.\n"
             "RULES:\n"
             "1) Answer ONLY using the provided CONTEXT.\n"
             "2) If the CONTEXT does not contain the answer, say exactly: "
             "\"I don't know based on the provided documents.\"\n"
             "3) Do not use general knowledge. Do not guess.\n"
             "4) Add a final line: EVIDENCE: <one sentence quoted or closely paraphrased from the CONTEXT>."),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
        ]
    )

    # LCEL: prompt → model → parser
    answer = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.1) | StrOutputParser()).invoke(
        {"question": question, "context": context}
    )

    # return answer and sources
    return {"answer": answer, "sources": sources}


#---------------------------------------------------------------------------------------------------------------------
# Baseline vs Multi-Query comparison
# Section: compare baseline vs multi-query
#---------------------------------------------------------------------------------------------------------------------

# Baseline: single retrieval using the original question
def answer_baseline(question: str, bm25: BM25Retriever, vec_retriever) -> dict:

    # get hybrid candidates for the question
    cand = hybrid_candidates(question, bm25, vec_retriever, fetch_k=25)

    # rerank and pick top 5
    top_docs, reason = llm_rerank(question, cand, top_n=5)
    if not top_docs:
        return {"answer": "I don't know based on the provided documents.\nEVIDENCE: (no supporting passage found)", "sources": [], "reason": reason}

    # if docs exist -> generate grounded answer
    out = grounded_answer(question, top_docs)
    out["reason"] = reason # attach rerank reason to output
    return out # return baseline output


# Multi-query: you generate several variants of the question and perform retrieval for each one
def answer_multiquery(question: str, bm25: BM25Retriever, vec_retriever) -> dict:

    # get list: original question + 3–5 rewrites
    queries = generate_multi_queries(question)

    merged = [] # merged candidates list across all rewrites
    seen = set() # set for deduplication
    for q in queries: # for each rewritten query
        cand = hybrid_candidates(q, bm25, vec_retriever, fetch_k=25) # retrieve candidates for that variant
        for d in cand: # iterate candidates
            key = (d.metadata.get("source", "unknown"), d.page_content[:200]) # deduplication key
            if key not in seen:
                seen.add(key)
                merged.append(d)

    # You rerank on the original question, but on a large pool with multi-query
    top_docs, reason = llm_rerank(question, merged, top_n=5)

    if not top_docs:
        return {"answer": "I don't know based on the provided documents.\nEVIDENCE: (no supporting passage found)", "sources": [], "reason": reason, "queries": queries}

    # if docs exist, generate grounded answer
    out = grounded_answer(question, top_docs)
    out["reason"] = reason
    out["queries"] = queries
    return out


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

    # vector retriever fetches up to 25 candidates
    vec_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    # build BM25 retriever from chunks
    bm25 = BM25Retriever.from_documents(chunks)

    print("Program ready (baseline vs multi-query). Type a question (or 'exit').")
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # compute baseline result
        base = answer_baseline(q, bm25, vec_retriever)

        # compute multi-query result
        mq = answer_multiquery(q, bm25, vec_retriever)

        print("\n=== BASELINE ===")
        print(base["answer"])
        print("Sources:", base["sources"])
        print("Rerank reason:", base.get("reason"))

        print("\n=== MULTI-QUERY ===")
        print("Queries:", mq.get("queries"))
        print(mq["answer"])
        print("Sources:", mq["sources"])
        print("Rerank reason:", mq.get("reason"))


if __name__ == "__main__":
    main()