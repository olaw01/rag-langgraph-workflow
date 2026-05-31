from __future__ import annotations

import logging # standard logging for workflow logs (debug/production).
from typing import List, Literal, TypedDict # types for graph state (TypedDict) and routing (Literal)

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END # state graph builder + START/END markers

from .rag import format_context # formats retrieved chunks into context and collects sources (rag.py)

# Module-level logger (src.app.graph)
logger = logging.getLogger(__name__)


# Define the state flowing through nodes (dict with fixed keys)
class RAGState(TypedDict):
    question: str                    # User question
    retrieved: List[dict]            # Retrieved passages {"text": str, "source": str}
    answer: str                      # LM-generated answer
    confidence: float                # Verifier confidence (0–1)
    decision: Literal["ok", "retry"] # Decision: accept or retry (routing)
    iteration: int                   # Loop iteration counter
    max_iterations: int              # Max attempts before stopping

# Pydantic schema returned by verifier
class VerifyResult(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)   # validate confidence is in [0,1]
    decision: Literal["ok", "retry"]            # model must choose “ok” or “retry”.
    rationale: str                              # short rationale (useful for debugging)


# Build LangGraph workflow with retriever, model and control thresholds
def build_graph_app(
    *,
    retriever,
    model: ChatOpenAI,
    verify_threshold: float,
    max_iterations: int,
):
    # Retrieve node: consumes state, returns a state patch
    def retrieve_node(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"]) # retrieval: retriever returns top-k Documents for the question.
        retrieved = [{"text": d.page_content, "source": d.metadata.get("source", "unknown")} for d in docs] # convert documents into simple dicts (easier for JSON/debug)
        return {"retrieved": retrieved, "iteration": state.get("iteration", 0) + 1} # update retrieved and increment iteration

    # Node: answer (grounded)
    # define prompt for answer generation
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounded Q&A assistant.\n"
                "Answer ONLY using the provided CONTEXT.\n"
                "If the answer is not in the context, say exactly: "
                "\"I don't know based on the provided documents.\"",
            ),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
        ]
    )
    # LCEL: prompt -> model -> string parser
    answer_chain = answer_prompt | model | StrOutputParser()

    # answer node: generates answer and returns a patch.
    def answer_node(state: RAGState) -> dict:
        context, _sources = format_context(state["retrieved"]) # build context from retrieved; _sources ignored here (context returns API)
        ans = answer_chain.invoke({"question": state["question"], "context": context}) # this is the LLM call (cost!)
        return {"answer": ans} # update answer


    # Node: verify (structured)
    # Verifier prompt: check whether answer is supported by context
    verify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", # System: strict verifier; choose ok/retry and return schema JSON.
                "You are a strict verifier.\n"
                "Given QUESTION, ANSWER and CONTEXT, decide if the answer is fully supported by context.\n"
                "If unsupported or missing key details, choose retry.\n"
                "Return JSON that matches the schema.",
            ),
            ("human", "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}"), # Human: provides question, answer, and context to evaluate
        ]
    )
    # enforce structured output as VerifyResult
    verifier = model.with_structured_output(VerifyResult)
    # LCEL: prompt → structured-output model
    verify_chain = verify_prompt | verifier

    # Verify node: returns confidence + decision.
    def verify_node(state: RAGState) -> dict:
        # Build context again (same one answer used)
        context, _sources = format_context(state["retrieved"])
        # Invoke verifier: get VerifyResult object
        out: VerifyResult = verify_chain.invoke(
            {"question": state["question"], "answer": state["answer"], "context": context}
        )
        # Start with model decision
        decision = out.decision
        # Hard threshold rule: if confidence < threshold -> force retry
        if float(out.confidence) < verify_threshold:
            decision = "retry"
        return {"confidence": float(out.confidence), "decision": decision}

    # Routing
    # Routing function decides: loop back or end
    def route_after_verify(state: RAGState) -> Literal["retrieve", "end"]:
        # If retry and under max -> loop to retrieve
        if state["decision"] == "retry" and state["iteration"] < state["max_iterations"]:
            return "retrieve"
        return "end"

    # Create a StateGraph with state type RAGState
    graph = StateGraph(RAGState)

    # Register the three nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)

    # Static edges: START -> retrieve -> answer -> verify
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")

    # Conditional edge after verify: loop or END.
    graph.add_conditional_edges("verify", route_after_verify, {"retrieve": "retrieve", "end": END})

    # Compile graph into an executable app
    app = graph.compile()

    # Helper runner (so API can call a simple function)
    # API calls run_graph(question) instead of manual state
    def run(question: str) -> dict:
        # initialize the starting state
        init_state: RAGState = {
            "question": question,
            "retrieved": [],
            "answer": "",
            "confidence": 0.0,
            "decision": "retry",
            "iteration": 0,
            "max_iterations": max_iterations,
        }
        # run graph synchronously (could also use async ainvoke)
        result = app.invoke(init_state)
        return result # return the final state result

    # build_graph_app returns the run_graph function injected into the API router
    return run