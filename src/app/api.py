from __future__ import annotations # Postpones annotation evaluation; useful for typing and Pydantic

import logging # standard logging for workflow logs (debug/production)
import uuid # Used to generate id -> request_id for each request (UUID)
from typing import List

from fastapi import APIRouter, HTTPException # APIRouter defines routes (endpoint); HTTPException raises HTTP errors
from pydantic import BaseModel, Field

# Module-level logger (src.app.api)
logger = logging.getLogger(__name__)

# Create a router to register endpoints
router = APIRouter()

# Request model for /query
class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000) # question must have  3–2000 chars -> input validation

# Response model for /query (shown in Swagger/OpenAPI)
class QueryResponse(BaseModel):

    answer: str             # Final answer
    sources: List[str]      # Unique sources list (e.g., .md file paths)
    confidence: float       # Confidence from verify node (0-1)
    iterations: int         # How many retrieve iterations were used (loop count)
    request_id: str         # Request ID for logging and debugging


# Factory function: takes run_graph(question) and creates routes (endpoints) using it
def make_router(run_graph):

    # FastAPI decorator: registers GET /health
    @router.get("/health")
    # Health endpoint handler.
    def health():
        return {"status": "ok"} # Return simple JSON -> used for monitoring/health checks

    @router.get("/")
    def root():
        return {"message": "Capstone RAG API. See /docs"}

    # POST /query endpoint; response_model enforces response shape (Pydantic)
    @router.post("/query", response_model=QueryResponse)

    # Handler takes request body as QueryRequest (FastAPI validates automatically)
    def query(req: QueryRequest):
        # Generate a unique request_id for logs and correlation
        request_id = str(uuid.uuid4())

        # Start error handling block -> failures become HTTP 500
        try:
            result = run_graph(req.question) # Call LangGraph workflow: retrieve -> answer -> verify(+loop), returns final state dict
            retrieved = result.get("retrieved", []) # Extract retrieved chunks list (or [] if missing)
            sources = sorted({x.get("source", "unknown") for x in retrieved}) # Collect unique sources from retrieved and sort (deterministic output)

            # Build response object matching QueryResponse schema
            resp = QueryResponse(
                answer=result.get("answer", ""),                    # Answer (fallback empty string if missing)
                sources=sources,                                    # Sources list
                confidence=float(result.get("confidence", 0.0)),    # Confidence (fallback 0.0), cast to float
                iterations=int(result.get("iteration", 0)),         # Iterations count (state key is iteration), cast to int
                request_id=request_id,                              # Attach request_id
            )

            # Log the success path
            # Log format: ID, iterations, confidence, sources count
            logger.info(
                "query_ok request_id=%s iterations=%s confidence=%.2f sources=%d",
                request_id,
                resp.iterations,
                resp.confidence,
                len(resp.sources),
            )
            return resp # Return response; FastAPI just change it to JSON

        # Catch-all exception -> prod can be more granular !!!!!!!!
        except Exception as e:
            logger.exception("query_failed request_id=%s error=%s", request_id, str(e))
            raise HTTPException(status_code=500, detail="Internal error")

    return router # Return the router with endpoints wired to run_graph