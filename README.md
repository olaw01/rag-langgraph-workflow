## Repository structure

This repo contains three layers of work:

- `basic_examples/`  
  Small, single-file examples used to learn individual concepts (async, prompts, etc.).  
  Think: quick experiments / scratchpad (not production).

- `mini_projects/`  
  Day-by-day mini projects (p01…p09) that demonstrate one concept end-to-end  
  (e.g., async API calling, LCEL, RAG baseline, MMR, hybrid retrieval + rerank, LangGraph workflow, MCP tools).

- `src/app/`  **Main project (Capstone)**
  The interview-ready application: a production-shaped RAG API built with FastAPI + LangGraph.
  This is the primary deliverable and the best entry point for reviewers.


## Capstone project (src/app)

The capstone is a FastAPI service that exposes:
- `GET /health` — health check
- `POST /query` — grounded RAG Q&A

Internally it runs a LangGraph workflow with conditional routing:
`retrieve → answer → verify`  
If the verifier confidence is low, the graph retries retrieval up to a configured limit, otherwise it returns the final answer.

Key features:
- RAG end-to-end (load → chunk → embed → store → retrieve → generate)
- strict grounding + "I don't know" fallback when docs do not support an answer
- structured verification (confidence + ok/retry decision)
- logging + env-based configuration (no secrets in code)
- unit + integration tests


## How it works (data flow)

Client -> POST /query
  -> LangGraph workflow:
     1) retrieve: fetch top-k chunks from vector store
     2) answer: generate answer using ONLY retrieved context
     3) verify: score support/confidence and decide ok/retry
  -> Response: answer + sources + confidence + iterations


## Running the capstone

### 1) Setup

```bash
python -m venv .venv
# activate venv
pip install -r requirements.txt
```
Create .env from .env.example and set OPENAI_API_KEY.

### 2) Run API

```bash
uvicorn src.app.main:app --reload
```

### 3) Example request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is LCEL?\"}"
```
