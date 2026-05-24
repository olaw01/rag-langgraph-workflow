import os
import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool


NOTES = [
    "LCEL = prompt | model | parser",
    "Tool = Python function available to the agent",
    "Agent = a model that can choose tools",
]


@tool
def search_notes(query: str) -> str:
    """Search local learning notes and return matching lines."""
    q = query.lower().strip()

    hits = []
    for note in NOTES:
        if q in note.lower():
            hits.append(note)

    if hits:
        return "\n".join(hits)

    return "No matches."


@tool
def calculate(expression: str) -> str:
    """Calculate a simple math expression. Only digits and + - * / ( ) are allowed."""
    expr = expression.strip()

    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "Error: unsupported characters."

    try:
        result = eval(expr, {"__builtins__": {}})
        return str(result)
    except Exception as error:
        return f"Error: {error}"


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[search_notes, calculate],
        system_prompt=(
            "You are a LangChain learning assistant.\n"
            "If your question is about notes, use search_notes.\n"
            "If the question requires counting, use calculate.\n"
        ),
    )

    question = "What is an agent? Use your notes."

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()