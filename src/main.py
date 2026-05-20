from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def build_chain():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Brak OPENAI_API_KEY w .env (w root projektu).")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Jesteś pomocnym asystentem. Odpowiadaj zwięźle."),
            ("human", "Wyjaśnij w 3 punktach: {topic}"),
        ]
    )

    return prompt | model | StrOutputParser()


def main() -> None:
    chain = build_chain()
    result = chain.invoke({"topic": "czym jest RAG + przykład zastosowania"})
    print(result)


if __name__ == "__main__":
    main()