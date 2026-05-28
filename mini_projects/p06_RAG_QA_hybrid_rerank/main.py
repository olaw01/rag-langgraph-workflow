import os
from pathlib import Path
from typing import List, Tuple # type hints

from dotenv import load_dotenv
from pydantic import BaseModel, Field # used to define a structured output schema for the reranker

# model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # model used for reranking and final answer generation, embedding model used to convert text into numerical vectors
from langchain_text_splitters import RecursiveCharacterTextSplitter # splits documents into smaller chunks
from langchain_core.prompts import ChatPromptTemplate # build prompts for the model
from langchain_core.output_parsers import StrOutputParser # converts the model response into a plain string

from langchain_community.document_loaders import DirectoryLoader, TextLoader # loads files from a directory, loads text files.
from langchain_community.vectorstores import Chroma # local vector database
from langchain_community.retrievers import BM25Retriever # keyword-based retriever (component that searches through large datasets or knowledge bases to find the most relevant information)


#---------------------------------------------------------------------------------------------------------------------
# LOADING
#
# This function loads all .md files from the data/docs folder.
# It converts them into LangChain Document objects.
#
# A Document contains:
# page_content = document text
# metadata = extra information, for example the source file path
#
# example:
# notes.md :
# LCEL means LangChain Expression Language.
# after load_documents:
# Document(
#     page_content="LCEL means LangChain Expression Language.",
#     metadata={"source": "data/docs/notes.md"}
# )
#---------------------------------------------------------------------------------------------------------------------


def load_documents(docs_dir: Path):
    """Load .md files from a folder into a list of Documents."""
    # Creates a loader object responsible for loading files from a directory
    loader = DirectoryLoader(
        str(docs_dir), # Converts Path into a string because DirectoryLoader expects a string path
        glob="**/*.md", # Find all .md files (including files inside subfolders)
        loader_cls=TextLoader, # Each found file should be loaded as a text file
        loader_kwargs={"encoding": "utf-8"}, # Sets file encoding to UTF-8, which is important for special characters
        show_progress=True, # Show loading progress
    )
    return loader.load() # Runs the loader and returns a list of documents.


#---------------------------------------------------------------------------------------------------------------------
# CHUNKING DOCUMENTS
#
# This function splits documents into smaller parts called chunks.
# Instead of passing a whole long document to the model, we split it into smaller searchable pieces.
#
# example:
# input
# LCEL means LangChain Expression Language.
# RAG means Retrieval-Augmented Generation.
# BM25 is keyword-based search.
# Chroma is a vector database.
# after chunking:
# [
#     Document(
#         page_content="LCEL means LangChain Expression Language.\n\nRAG means Retrieval-Augmented Generation.",
#         metadata={"source": "notes.md"}
#     ),
#     Document(
#         page_content="BM25 is keyword-based search.\n\nChroma is a vector database.",
#         metadata={"source": "notes.md"}
#     )
#---------------------------------------------------------------------------------------------------------------------


def chunk_documents(docs):
    """Split documents into chunks for retrieval."""
    # Creates a splitter, which is a tool for splitting text.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # One chunk should have around 800 characters (not words!)
        chunk_overlap=120, # Around 120 characters are repeated between chunks (this prevents losing context at chunk boundaries)
        separators=["\n\n", "\n", ". ", " ", ""], # This is the order of separators used to split text (first tries to split by paragraphs, then lines, then sentences, then spaces, and finally characters)
    )
    return splitter.split_documents(docs) # Splits all documents and returns a list of chunks


#---------------------------------------------------------------------------------------------------------------------
# BUILD VECTORSTORE
#
# This function creates a local Chroma vector store.
# It takes chunks, converts each chunk into an embedding, and stores them on disk.
# An embedding is a numerical representation of text meaning.
#
# example
# "Chroma is a local vector database."
# embedding
# [0.123, -0.441, 0.882, 0.015, ...]
#
# Chroma:
# text: "LCEL means LangChain Expression Language."
# embedding: [0.12, -0.41, 0.77, ...]
# metadata: {"source": "notes.md"}
#---------------------------------------------------------------------------------------------------------------------


def build_vectorstore(chunks, persist_dir: Path):
    """Create a local Chroma vector store on disk."""

    # Creates an embedding model object. The model (text-embedding-3-small converts) text into numerical vectors
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Creates a Chroma database from documents/chunks
    return Chroma.from_documents(
        documents=chunks, # Passes chunks into Chroma
        embedding=embeddings, # Tells Chroma how to convert text into embeddings
        persist_directory=str(persist_dir), # Defines where Chroma should save data on disk
        collection_name="p06_docs", # Collection name in Chroma - it is like a group name for stored documents
    )


#---------------------------------------------------------------------------------------------------------------------
# FORMAT CONTEXT
#
# This function takes retrieved documents/chunks and converts them into one context string for the model.
# It also collects unique source file names.
#
# INPUT:
#
# docs = [
#     Document(
#         page_content="LCEL means LangChain Expression Language.",
#         metadata={"source": "notes.md"}
#     ),
#     Document(
#         page_content="RAG means Retrieval-Augmented Generation.",
#         metadata={"source": "notes.md"}
#     )
# ]
#
#
# OUTPUT:
#
# context = """
# [source=notes.md]
# LCEL means LangChain Expression Language.
#
# ---
#
# [source=notes.md]
# RAG means Retrieval-Augmented Generation.
# """
#
# sources = ["notes.md"]
#
#---------------------------------------------------------------------------------------------------------------------


def format_context(docs) -> Tuple[str, List[str]]:
    """Combine docs into a context string and collect unique sources."""

    sources = [] # Creates an empty list for sources
    parts = [] # Creates an empty list for text parts
    for d in docs: # Loops through every document/chunk
        src = d.metadata.get("source", "unknown") # Gets the document source from metadata - if there is no source, uses "unknown"
        sources.append(src) # Adds source to the sources list
        parts.append(f"[source={src}]\n{d.page_content}") # Adds chunk text together with its source
    return "\n\n---\n\n".join(parts), sorted(set(sources)) # context is one joined text, sources is a sorted list of unique sources


#---------------------------------------------------------------------------------------------------------------------
# HYBRID RETRIEVAL
# Hybrid candidates: BM25 + Vector
#
# This function performs hybrid retrieval.
# It retrieves candidate chunks using two methods:
# 1. BM25 — keyword search.
# 2. Vector retriever — semantic search.
# Then it merges the results and removes duplicates.
#
# BM25 is useful when the question contains exact words, acronyms, names, or IDs.
# Example:
# What is LCEL?
# BM25 finds a chunk containing LCEL.
#
# Vector retriever is useful when the question uses different words but has similar meaning.
#---------------------------------------------------------------------------------------------------------------------


def hybrid_candidates(question: str, bm25: BM25Retriever, vec_retriever, fetch_k: int = 25):
    """
    Return a merged, de-duplicated candidate list from:
    - BM25 (keyword/sparse) for exact phrases/acronyms/IDs
    - Vector (dense) for semantic similarity
    """
    # BM25 candidates (keyword) / Sets how many results BM25 should return.
    bm25.k = fetch_k
    bm25_docs = bm25.invoke(question) # Runs BM25 search for the user question - Returns keyword-matching chunks

    # Vector candidates (semantic)
    vec_docs = vec_retriever.invoke(question) # Runs vector search for the user question - Returns semantically similar chunks

    # De-duplicate using a simple heuristic (source + prefix)
    seen = set() # seen remembers which chunks were already added
    merged = [] # merged stores final combined results.
    for d in bm25_docs + vec_docs: # Loops through BM25 results first, then vector results
        key = (d.metadata.get("source", "unknown"), d.page_content[:200]) # Creates a simple duplicate-detection key (source + first 200 characters of content)
        if key not in seen: # Checks if this chunk was already added
            seen.add(key)
            merged.append(d)
    return merged # Returns the merged candidate list


#---------------------------------------------------------------------------------------------------------------------
# LLM RERANKING (no Cohere needed) - Pydantic schema
# This block performs reranking using an LLM, without Cohere.
# RerankPick is a structured output schema for the reranker. Instead of free-form text, we force the model to return:
#
# - indices: selected candidate indices (can be empty)
# - reason: a short rationale
#
# This makes downstream code reliable because you can safely select docs[i].
#---------------------------------------------------------------------------------------------------------------------


# Define a Pydantic schema (“contract”) for the reranker output
class RerankPick(BaseModel):
    indices: List[int] = Field(description="Selected indices (can be empty).") # indices is the list of candidate selected by the reranker (can be empty)
    reason: str = Field(description="Short reason for the selection.") # reason is a short rationale (useful for logging/debugging)

# Reranking function: takes the question and candidate docs, selects up to top_n
def llm_rerank(question: str, docs, top_n: int = 5):
    """
    Ask the LLM to pick the best passages among candidates.
    If none contains the answer, it should return an empty list of indices.
    """

    items = [] # You create a list of candidate for the prompt

    for i, d in enumerate(docs): # enumerate candidates and assign each an index i
        snippet = d.page_content[:500].replace("\n", " ") # take first 500 chars and delete newlines to keep the prompt compact
        src = d.metadata.get("source", "unknown") # read the source (e.g., file path) from metadata
        items.append(f"{i}) source={src} text={snippet}") # append a candidate line: “index + source + snippet”


    # build a reranker prompt using system/human messages
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

    # create the reranker model with temperature=0.0 and enforce structured output RerankPick
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(RerankPick)

    # LCEL: prompt -> model. Output will be a RerankPick object (not plain text)
    chain = prompt | model

    # invoke the chain, filling placeholders. Candidates are joined into one newline-separated block
    out: RerankPick = chain.invoke({"question": question, "candidates": "\n".join(items)})

    # map indices back to Document
    chosen = [docs[i] for i in out.indices if 0 <= i < len(docs)]

    # Return selected docs + the reason (for logging)
    return chosen, out.reason


#---------------------------------------------------------------------------------------------------------------------
# FINAL GROUNDED ANSWER
#
# From here: final answer generation, grounded only in context !!!!!!!!!!!!
#---------------------------------------------------------------------------------------------------------------------

# generates an answer using only docs (the reranked top docs)
def rag_answer(question: str, docs) -> dict:

    # format the chunks into a single context string and collect sources list
    context, sources = format_context(docs)

    # build the final grounded Q&A prompt
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

    # set answering model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    # LCEL: prompt -> model -> parse output into a plain string
    chain = prompt | model | StrOutputParser()

    # invoke the chain with question and context. This is when the API call happens
    answer = chain.invoke({"question": question, "context": context})

    # return the answer plus the list of sources
    return {"answer": answer, "sources": sources}


def main():

    # loads .env variables into environment
    load_dotenv()
    # check if API key is present
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    base_dir = Path(__file__).parent # base_dir is the directory where this main.py exist
    docs_dir = base_dir / "data" / "docs" # input docs path
    persist_dir = base_dir / "data" / "chroma" # where Chroma persists the vector index (should be .gitignored)

    # load .md files
    docs = load_documents(docs_dir)

    # split docs into chunks for retrieval
    chunks = chunk_documents(docs)

    # build a Chroma vector store with chunk embeddings
    vectorstore = build_vectorstore(chunks, persist_dir)

    # create a semantic retriever fetching up to 25 candidate chunks
    vec_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    # create a BM25 retriever (keyword / exact term matching)
    bm25 = BM25Retriever.from_documents(chunks)

    print("Hybrid + LLM rerank RAG is ready. Type a question (or 'exit').")

    # interactive loop to accept many questions
    while True:

        # read a user query and trim whitespace
        q = input("\n> ").strip()

        # exit condition
        if q.lower() in {"exit", "quit"}:
            break

        # hybrid retrieval: BM25 + vector candidates (fetch_k=25 for BM25)
        candidates = hybrid_candidates(q, bm25=bm25, vec_retriever=vec_retriever, fetch_k=25)
        # LLM reranker selects top 5 passages and returns a reason
        top_docs, reason = llm_rerank(q, candidates, top_n=5)

        # If reranker found nothing, do not hallucinate
        if not top_docs:
            print("\nAnswer:\nI don't know based on the provided documents.\nEVIDENCE: (no supporting passage found)")
            print("\nSources:\n(none)")
            continue

        # if you have top docs -> generate the final grounded answer
        result = rag_answer(q, top_docs)
        print("\n[rerank reason]", reason)
        print("\nAnswer:\n", result["answer"])
        print("\nSources:\n", "\n".join(result["sources"]))


if __name__ == "__main__":
    main()