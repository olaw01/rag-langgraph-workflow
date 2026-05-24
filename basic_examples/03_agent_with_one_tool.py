from langchain.agents import create_agent
from langchain_core.tools import tool
import os
from dotenv import load_dotenv


@tool
def double_number(number: int) -> str:
    """Double a given integer number."""
    result = number * 2
    return str(result)

def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")


    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[double_number],
        system_prompt=(
            "You are a simple math assistant. "
            "If a user asks you to double a number, use the double_number tool. "
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Double the number 21.",
                }
            ]
        }
    )

    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()