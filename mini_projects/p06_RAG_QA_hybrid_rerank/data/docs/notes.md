\## LCEL



LCEL means LangChain Expression Language.

It is a pipeline style where you connect components like this:



prompt | model | parser



\## Structured Output



Structured output means that the model returns data in a specific schema.

For example, you can use Pydantic to define the expected fields.



Structured output is useful because it is more reliable than asking the model to return plain text.



\## Asyncio



asyncio.gather runs many async tasks concurrently.



Semaphore limits concurrency.

It controls how many tasks can run inside a selected block of code at the same time.



\## RAG



RAG means Retrieval-Augmented Generation.



In RAG, the system first retrieves relevant documents or chunks.

Then the model generates an answer using only the retrieved context.

