#---------------------------------------------------------------------------------------------------------------------
# retrieve -> searching for a matching chunk
# question → find relevant chunks → answer based on chunks
#---------------------------------------------------------------------------------------------------------------------

chunks = [
    "LCEL means LangChain Expression Language. It uses prompt | model | parser.",
    "Structured output uses a schema like Pydantic.",
    "Asyncio gather runs many tasks concurrently. Semaphore limits concurrency.",
]

question = "What limits concurrency?"

for chunk in chunks:
    if "concurrency" in chunk.lower():
        print("Found chunk:")
        print(chunk)