#---------------------------------------------------------------------------------------------------------------------
# INSTALLATION
# python -m pip install langchain-community chromadb langchain-text-splitters
# python -m pip install langchain langchain-openai python-dotenv
#---------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------
# RAG Fundamentals
#
# User question
# → search in documents
# → get relevant chunks
# → give chunks to model
# → answer based on chunks
#
#
# Key concepts:
#
# - RAG = Retrieval-Augmented Generation.
# - RAG pipeline = load → chunk → embed → store → retrieve → generate.
# - Documents are split into smaller chunks before indexing.
# - Embeddings convert text into vectors, so we can search by meaning.
# - Vector stores like Chroma store chunks together with their embeddings and metadata.
# - Retriever finds the most relevant chunks for a user question.
# - The model generates an answer using retrieved context.
# - Sources can be shown using document metadata, for example `metadata["source"]`.
#
# Important notes:
#
# - Chunk size controls how much text is inside one chunk.
# - Chunk overlap helps prevent losing information at chunk boundaries.
# - "Lost in the middle" means that models may miss information placed in the middle of a long context.
# - A good chunk size and overlap depend on the document type, task, and retrieval quality.
#---------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma


def load_documents(docs_dir: Path):
    """
    Load documents from the docs directory. (mini_projects/p04_RAG_QA_similarity/data/docs)
    """
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    return loader.load()


#---------------------------------------------------------------------------------------------------------------------
# WITHOUT OVERLAP
# Chunk 1:
# Semaphore limits
# Chunk 2:
# concurrency in async code.
#
# WITH OVERLAP
# Chunk 1:
# Semaphore limits concurrency
# Chunk 2:
# Semaphore limits concurrency in async code.
#---------------------------------------------------------------------------------------------------------------------

def chunk_documents(documents):
    """
    Split documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


#---------------------------------------------------------------------------------------------------------------------
# Chroma is a local vector database.
#
# This is where we store:
#
# chunk text + embedding + metadata
#
# More or less:
# - chunk text
# - its numeric vector
# - source, e.g.: notes.md
#---------------------------------------------------------------------------------------------------------------------

def build_vectorstore(chunks, persist_dir: Path):
    """
    Create a Chroma vector database from document chunks.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # This converts each chunk into numbers

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="mp04_docs",
    )

    return vectorstore


#---------------------------------------------------------------------------------------------------------------------
# 1. context - one large text that the model then receives
# 2. unique_sources - a list of unique sources that you then show to the user
#---------------------------------------------------------------------------------------------------------------------

def format_context(documents) -> Tuple[str, List[str]]:
    """
    Convert retrieved documents into one context string
    and collect unique source file names.
    """
    context_parts = []
    sources = []

    for document in documents:
        source = document.metadata.get("source", "unknown")
        content = document.page_content

        sources.append(source)
        context_parts.append(f"[source={source}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    unique_sources = sorted(set(sources))

    return context, unique_sources


def rag_answer(question: str, retriever) -> dict:
    """
    Retrieve relevant chunks and generate an answer based on them.
    """
    retrieved_documents = retriever.invoke(question) # this returns a list of documents/chunks

    context, sources = format_context(retrieved_documents) # take the found chunks and put them together into one text for the model

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a Q&A assistant.\n"
             "RULES:\n"
             "1) Answer ONLY based on CONTEXT.\n"
             "2) If the context doesn't provide an answer, state precisely: "
             "'I don't know based on the documents provided.'\n"
             "3) Don't guess or add facts.\n"
             "4) Keep your answer short and to the point."
             ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}",
            ),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    chain = prompt | model | StrOutputParser() # LCEL

    answer = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": answer,
        "sources": sources,
    }


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env file")

    # dir paths
    base_dir = Path(__file__).parent #C:\Users\awawr\PythonProject\PythonProject\rag-langgraph-workflow\mini_projects\p04_RAG_QA_similarity
    docs_dir = base_dir / "data" / "docs"
    persist_dir = base_dir / "data" / "chroma"

    # load documents form path
    documents = load_documents(docs_dir)

    # chunk documents for small part of information
    chunks = chunk_documents(documents)

    # build a local vector datastore (chroma)
    vectorstore = build_vectorstore(chunks, persist_dir)

    # ---------------------------------------------------------------------------------------------------------------------
    # Retriever is the element that receives the question and finds the matching fragments
    # SIMILARITY
    # retriever = vectorstore.as_retriever(
    #     search_kwargs={
    #         "k": 4, # return the 4 most matching chunks !!!
    #     }
    # )
    # MMR
    # fetch_k=20 -> fetch 20 candidates by similarity
    # k=4 -> select the final 4, but diverse ones
    # ---------------------------------------------------------------------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

    print("RAG Q&A is ready.")
    print("Type your question or type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ")

        if question.strip().lower() in {"exit", "quit"}:
            break

        result = rag_answer(question, retriever)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    main()



