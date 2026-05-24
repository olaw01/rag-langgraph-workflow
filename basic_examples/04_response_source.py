chunks = [
    {
        "content": "LCEL means LangChain Expression Language.",
        "source": "notes.md",
    },
    {
        "content": "Semaphore limits concurrency.",
        "source": "async_notes.md",
    },
]

question = "What does Semaphore do?"

for chunk in chunks:
    if "Semaphore" in chunk["content"]:
        print("Answer based on:")
        print(chunk["content"])
        print("Source:")
        print(chunk["source"])