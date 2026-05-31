#---------------------------------------------------------------------------------------------------------------------
# Run app (terminal):
# uvicorn src.app.main:app --reload
#---------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI # import FastAPI class to create an HTTP server (ASGI)

from langchain_openai import ChatOpenAI

from .settings import Settings # import settings (env-based config -> settings.py) to avoid hardcoding configuration
from .logging_config import configure_logging # import logging configuration (logging_config .py)
from .rag import load_documents, chunk_documents, build_vectorstore # import RAG helpers: load docs, chunk them, and build the vector store (rag.py)
from .graph import build_graph_app # import LangGraph workflow builder; returns run_graph(question) (graph.py)
from .api import make_router # import FastAPI router factory (/health and /query endpoints) (api.py)

#---------------------------------------------------------------------------------------------------------------------


# builds the FastAPI app and all dependencies (RAG + LangGraph)
def create_app() -> FastAPI:
    # instantiate Settings — loads values from .env and environment
    settings = Settings()
    # configure logging (format + level) for readable console logs
    configure_logging(settings.log_level)

    # Ensure OpenAI key is available for langchain-openai
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # base working directory (where run uvicorn) -> repo root
    base_dir = Path(".")
    docs_dir = (base_dir / settings.docs_dir).resolve()
    chroma_dir = (base_dir / settings.chroma_dir).resolve()
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # load .md files from data/docs into LangChain Document objects
    docs = load_documents(docs_dir)
    # split documents into chunks — better for retrieval.
    chunks = chunk_documents(docs)

    # build Chroma vector store: embed chunks and persist the index to disk
    vectorstore = build_vectorstore(
        chunks, chroma_dir, settings.embedding_model, collection_name="capstone_docs"
    )

    # create retriever that returns top-k relevant chunks for a query
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})

    # create chat model for answer/verify; temp=0.1 for stable outputs
    model = ChatOpenAI(model=settings.model_name, temperature=0.1)

    # build LangGraph workflow and get run_graph(question) -> result dict
    run_graph = build_graph_app(
        retriever=retriever,
        model=model,
        verify_threshold=settings.verify_threshold,
        max_iterations=settings.max_iterations,
    )

    # create FastAPI app with metadata
    app = FastAPI(title="Capstone RAG API", version="1.0.0")

    # mount routes (/health, /query) that call run_graph to handle queries
    app.include_router(make_router(run_graph))

    # return the fully configured FastAPI app
    return app

# Uvicorn imports this module and needs `app` -> uvicorn src.app.main:app --reload
app = create_app()