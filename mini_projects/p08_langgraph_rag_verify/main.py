import os
from pathlib import Path
from typing import TypedDict, List, Literal # typing for graph state and routing

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate # prompts as system/human messages
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, START, END  # build a state graph + START/END markers.
from langgraph.checkpoint.memory import MemorySaver # In-memory checkpointing (RAM)


#---------------------------------------------------------------------------------------------------------------------
# 1) Graph State (what flows through the graph)
#---------------------------------------------------------------------------------------------------------------------

# defines the “state” that flows through the graph -> state type as a dict with fixed keys
class RAGState(TypedDict):
    question: str                    # user question
    retrieved: List[dict]            # retrieved chunks as list of {"text": str, "source": str}
    answer: str                      # generated answer
    confidence: float                # confidence score from verify
    decision: Literal["ok", "retry"] # routing decision
    iteration: int                   # loop counter - how many retrieve cycles
    max_iterations: int              # maximum iterations before stopping


#---------------------------------------------------------------------------------------------------------------------
# 2) Helpers: load -> chunk -> vectorstore (outside the graph)
# Helper functions for index building - not graph nodes
#---------------------------------------------------------------------------------------------------------------------

# loads documents from docs_dir
def load_documents(docs_dir: Path):
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader, # read each file as text
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load() # returns a list of Documents

# splits documents into chunks
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120, # overlap between chunks to preserve boundary context
        separators=["\n\n", "\n", ". ", " ", ""], # split priority: paragraphs -> lines -> sentences -> words -> chars
    )
    return splitter.split_documents(docs) # returns chunked Documents

# builds a Chroma vector store and persists it.
def build_vectorstore(chunks, persist_dir: Path):
    # embedding model to convert chunk text into vectors.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # create an index from chunk documents
    return Chroma.from_documents(
        documents=chunks,                       # input docs: chunks
        embedding=embeddings,                   # embedding function
        persist_directory=str(persist_dir),     # persistence folder (should be gitignored) - Chroma !
        collection_name="p08_docs",
    )

# Combines downloaded fragments into one “context” for LLM
def format_context(retrieved: List[dict]) -> str:

    parts = [] # parts of the context

    # iterate over retrieved chunks.
    for item in retrieved:
        parts.append(f"[source={item['source']}]\n{item['text']}") # append the chunk plus its source
    return "\n\n---\n\n".join(parts) # join chunks with separators


#---------------------------------------------------------------------------------------------------------------------
# 3) Node: retrieve
# First node: retrieve context
#---------------------------------------------------------------------------------------------------------------------

# Node factory - returns a node function capturing the retriever
def make_retrieve_node(retriever):

    # actual node - takes state and returns a state patch
    def retrieve_node(state: RAGState) -> dict:
        # read the question from state
        q = state["question"]

        docs = retriever.invoke(q)  # LangChain Runnable interface -> retrieval happens here via Runnable API

        retrieved = [] # prepare a list of simple dict results

        # iterate through returned Documents
        for d in docs:
            # append each chunk to the list
            retrieved.append(
                {
                    "text": d.page_content, # chunk text
                    "source": d.metadata.get("source", "unknown"),
                }
            )

        # node returns a state patch
        return {
            "retrieved": retrieved, # update retrieved chunks
            "iteration": state.get("iteration", 0) + 1, # increment iteration counter
        }

    return retrieve_node # returns the node function


#---------------------------------------------------------------------------------------------------------------------
# 4) Node: answer (grounded)
# Factory for answer node, close model and prompt.
#---------------------------------------------------------------------------------------------------------------------

# build the Q&A prompt
def make_answer_node(model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grounded Q&A assistant.\n"
                "Answer ONLY using the provided CONTEXT.\n"
                "If the answer is not in the context, say exactly: "
                "\"I don't know based on the provided documents.\""
            ),
            ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
        ]
    )

    # LCEL: prompt -> model -> string
    chain = prompt | model | StrOutputParser()

    # Node: generates an answer from current state
    def answer_node(state: RAGState) -> dict:
        context = format_context(state["retrieved"]) # build context from retrieved chunks
        ans = chain.invoke({"question": state["question"], "context": context}) # invoke LLM and get the answer
        return {"answer": ans} # update answer field in state

    return answer_node # return the node function


#---------------------------------------------------------------------------------------------------------------------
# 5) Node: verify (confidence + route decision)
#---------------------------------------------------------------------------------------------------------------------

# verifier output schema
class VerifyResult(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="0 to 1 confidence") # enforce 0–1 range
    decision: Literal["ok", "retry"] = Field(description="ok if grounded & sufficient, else retry") # verifiy chooses ok or retry
    rationale: str = Field(description="Short reason") # short rationale


# Factory for verify node; threshold is the minimum confidence
def make_verify_node(model: ChatOpenAI, threshold: float = 0.7):

    # prompt for verifier
    verify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict verifier.\n"
                "Given QUESTION, ANSWER and CONTEXT, decide if the answer is fully supported by context.\n"
                "If unsupported or missing key details, choose retry.\n"
                "Return JSON that matches the schema."
            ),
            ("human", "QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}"),
        ]
    )
    # enforce VerifyResult structured output
    verifier = model.with_structured_output(VerifyResult)
    verify_chain = verify_prompt | verifier

    # verify node evaluates question/answer/context
    def verify_node(state: RAGState) -> dict:
        context = format_context(state["retrieved"]) # build context string
        # get structured output: confidence + decision + rationale
        out: VerifyResult = verify_chain.invoke(
            {"question": state["question"], "answer": state["answer"], "context": context}
        )

        # Extra rule: even if model says ok, enforce threshold
        decision = out.decision

        # If confidence is below threshold
        if out.confidence < threshold:
            decision = "retry" # force retry regardless of model suggestion

        # update confidence and decision in state
        return {
            "confidence": float(out.confidence),
            "decision": decision,
        }

    return verify_node # return the node


#---------------------------------------------------------------------------------------------------------------------
# 6) Routing function for conditional edge after verify
#---------------------------------------------------------------------------------------------------------------------
# routing function: decides next step after verify
def route_after_verify(state: RAGState) -> Literal["retrieve", "end"]:
    # if retry and we have not hit max attempts
    if state["decision"] == "retry" and state["iteration"] < state["max_iterations"]:
        return "retrieve" # loop back to retrieve -> back
    return "end" # otherwise end


#---------------------------------------------------------------------------------------------------------------------
# 7) Main: build graph + run
#---------------------------------------------------------------------------------------------------------------------

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    base_dir = Path(__file__).parent
    docs_dir = base_dir / "data" / "docs"
    persist_dir = base_dir / "data" / "chroma"

    # Build index (load -> chunk -> embed -> store)
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, persist_dir)

    # Retriever and model
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Nodes (closures capture retriever/model)
    retrieve_node = make_retrieve_node(retriever)
    answer_node = make_answer_node(model)
    verify_node = make_verify_node(model, threshold=0.7)

    # Graph definition
    graph = StateGraph(RAGState) # Create LangGraph state graph

    graph.add_node("retrieve", retrieve_node)   # register retrieve node
    graph.add_node("answer", answer_node)       # register answer node
    graph.add_node("verify", verify_node)       # register verify node

    graph.add_edge(START, "retrieve")                   # START goes to retrieve
    graph.add_edge("retrieve", "answer")        # retrieve -> answer
    graph.add_edge("answer", "verify")          # answer -> verify

    # add conditional routing edges from verify
    graph.add_conditional_edges(
        "verify",             # routing source node
        route_after_verify,          # function that returns “retrieve” or “end”
        {
            "retrieve": "retrieve",  # loop - map routing labels to destinations
            "end": END,              # finish - end conditional edges
        },
    )

    # Dev checkpointing: stores state in memory by thread_id
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer) # compile the graph into an executable app

    # Run
    question = input("Ask a question: ").strip()

    # Initial graph state
    initial_state: RAGState = {
        "question": question,
        "retrieved": [],
        "answer": "",
        "confidence": 0.0,
        "decision": "retry",
        "iteration": 0,
        "max_iterations": 3,
    }

    # run the graph. thread_id identifies the checkpoint thread in dev
    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "dev-thread-1"}},
    )

    print("\n=== FINAL ===")
    print("Answer:\n", result["answer"])
    print("\nConfidence:", result["confidence"])
    print("\nDecision:", result["decision"])
    print("\nIterations used:", result["iteration"])
    print("\nSources:")
    for item in result["retrieved"]:
        print("-", item["source"])


if __name__ == "__main__":
    main()