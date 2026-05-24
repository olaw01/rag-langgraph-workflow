#---------------------------------------------------------------------------------------------------------------------
# Goal:
# - Create a simple LangChain agent
# - Give it two tools:
# 1) search_notes - Search local notes
# 2) calculate - Calculate simple operations
# - Force a structured response via Pydantic
#---------------------------------------------------------------------------------------------------------------------
import os
import re # re is a module for regular expressions, or regexes

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool


#---------------------------------------------------------------------------------------------------------------------
# MINI KNOWLEDGE BASE, OR LOCAL NOTES
# For now, we're not using PDFs, embeddings, or a vector database. We simply have a list of strings.
#---------------------------------------------------------------------------------------------------------------------

NOTES = [
    "LangChain LCEL = prompt | model | parser",
    "Structured output = contract enforced with Pydantic; reliability > formatting",
    "Asyncio: gather = run tasks concurrently, Semaphore = limit concurrency",
]


#---------------------------------------------------------------------------------------------------------------------
# TOOL NUMBER 1: search_notes
# @tool - tells LangChain: "This function can be used by the agent as a tool"
#---------------------------------------------------------------------------------------------------------------------

@tool
def search_notes(query: str) -> str:
    """Search short local notes for a query and return matching lines."""

    # Take the query text, e.g., "LCEL"
    # lower() converts to lowercase: "lcel"
    # strip() removes spaces: "lcel"
    q = query.lower().strip()
    print(f"Searching notes for {q}")

    # add n for each n from NOTES if q is in n.lower() / list comprehension
    hits = [n for n in NOTES if q in n.lower()]

    # empty list in python : false
    # If the hits list is not empty, return the found notes joined with new lines.
    # If the hits list is empty, return the text "No matches.".
    return "\n".join(hits) if hits else "No matches."


#---------------------------------------------------------------------------------------------------------------------
# TOOL NUMBER 2: calculate
#---------------------------------------------------------------------------------------------------------------------

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression like '2+2' or '10*(3-1)'. Digits and +-*()/ only."""

    # We remove unnecessary spaces from the beginning and end.
    expr = expression.strip()

    # We check if the operation contains only allowed characters:
    # digits, period, plus, minus, multiplication, division, parentheses, and spaces.
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "Error: unsupported characters."
    # eval counts text as a Python expression
    # {"__builtins__": {}} restricts access to built-in Python functions.
    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


#---------------------------------------------------------------------------------------------------------------------
# AGENT RESPONSE SCHEMA
#---------------------------------------------------------------------------------------------------------------------

class AgentAnswer(BaseModel):

    # The final answer from the agent.
    answer: str = Field(description="Final answer for the user.")

    # A list of tool names used by the agent.
    #
    # default_factory=list means:
    # "If the value is not provided, create a new empty list."

    # noinspection PyDataclass
    used_tools: list[str] = Field(
        default_factory=list,
        description="Names of tools used by the agent.",
    )


#---------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION
#---------------------------------------------------------------------------------------------------------------------
def main() -> None:
    # Load environment variables from .env
    load_dotenv()

    # Check if OPENAI_API_KEY exists.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    # Create the agent.
    # The agent gets:
    # - model: the LLM brain
    # - tools: functions it can use
    # - system_prompt: rules for behavior
    # - response_format: structured output schema !!!
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_notes, calculate],
        system_prompt=(
            "You are a technical assistant.\n"
            "RULES:\n"
            "1) If the question is related to notes, ALWAYS use the search_notes tool.\n"
            "2) If the question requires calculation, ALWAYS use the calculate tool.\n"
            "3) In the used_tools field, include the list of tool names you used.\n"
        ),
        response_format=AgentAnswer,
    )

    # First question:
    # This should trigger the search_notes tool.
    out1 = agent.invoke({"messages": [{"role": "user", "content": "Explain LCEL in one sentence. Use the notes."}]})
    print("\n=== Q1 structured_response ===")
    print(out1["structured_response"])

    # Second question:
    # This should trigger the calculate tool.
    out2 = agent.invoke({"messages": [{"role": "user", "content": "Calculate 10*(3-1) + 4."}]})
    print("\n=== Q2 structured_response ===")
    print(out2["structured_response"])



if __name__ == "__main__":
    main()