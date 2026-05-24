#---------------------------------------------------------------------------------------------------------------------
# Documents can be long, so we don't put the entire book into the model.
# We cut the document into smaller pieces -> chunks.
#---------------------------------------------------------------------------------------------------------------------


text = """
LCEL means LangChain Expression Language.
It allows you to build chains using the pipe operator.

Structured output uses a schema like Pydantic.
It makes model responses more reliable.

Asyncio gather runs many tasks concurrently.
Semaphore limits how many tasks run at the same time.
"""

chunks = text.split("\n\n")

for i, chunk in enumerate(chunks):
    print(f"CHUNK {i}:")
    print(chunk)
    print()