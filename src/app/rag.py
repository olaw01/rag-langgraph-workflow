from __future__ import annotations

from pathlib import Path
from typing import List, Tuple # helper types List/Tuple

from langchain_openai import OpenAIEmbeddings # OpenAI embeddings wrapper (text -> vectors)
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splitter for chunking docs into overlapping chunks
from langchain_community.document_loaders import DirectoryLoader, TextLoader # DirectoryLoader scans a folder; TextLoader reads a single file as text
from langchain_community.vectorstores import Chroma # Chroma is a local vector store (disk persistence) for chunk embedding


def load_documents(docs_dir: Path):
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    return loader.load()


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, chroma_dir: Path, embedding_model: str, collection_name: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
    )

# formats retrieved chunks into a single context string + returns sources list
def format_context(retrieved: List[dict]) -> Tuple[str, List[str]]:
    parts = [] # collect chunk texts (with source headers) here
    sources = [] # collect sources (e.g., file paths) to return via API
    for item in retrieved: # ierate over chunk dicts: {"text": ..., "source": ...}
        src = item.get("source", "unknown") # read source; fallback to "unknown" to avoid crashing
        sources.append(src) # append source (later  via set() makes them unique).
        parts.append(f"[source={src}]\n{item.get('text','')}") # append chunk to context with source header (
    context = "\n\n---\n\n".join(parts) # join chunks with separators so the model sees boundaries
    return context, sorted(set(sources)) # return context and unique sources (sorted for deterministic output)