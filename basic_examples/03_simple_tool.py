from langchain_core.tools import tool

# @tool - turns a function into a tool
@tool
def say_hello(name: str) -> str:
    """Return a short greeting for a given name."""
    return f"Hello, {name}!"

def main() -> None:

    print("Tool name:")
    print(say_hello.name)

    print("\nTool description:")
    print(say_hello.description)

    print("\nManual tool call:")
    result = say_hello.invoke({"name": "Ola"}) # it is a manual call to tools witout agent
    print(result)

# terminal: python *file path*
if __name__ == "__main__":
    main()