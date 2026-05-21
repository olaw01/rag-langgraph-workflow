from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class TopicSummary(BaseModel):
    topic: str = Field(description="Temat, który użytkownik podał")
    bullets: List[str] = Field(description="3 zwięzłe punkty", min_length=3, max_length=3)
    risks: List[str] = Field(description="Potencjalne ryzyka / pułapki", default_factory=list)
    next_steps: List[str] = Field(description="Najbliższe kroki do zrobienia", default_factory=list)


def build_structured_chain():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Brak OPENAI_API_KEY w .env (w root projektu).")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Zwracaj odpowiedź WYŁĄCZNIE w formacie zgodnym ze schematem."),
            ("human", "Zrób podsumowanie tematu: {topic}. Daj 3 punkty, ryzyka i next steps."),
        ]
    )

    # model ma zwrócić zgodny obiekt Pydantic
    structured_model = model.with_structured_output(TopicSummary)

    return prompt | structured_model


def main() -> None:
    chain = build_structured_chain()

    result: TopicSummary = chain.invoke({"topic": "RAG w LangChain (dla początkującego)"})
    # result to OBIEKT, nie string
    print(result)
    print("\n--- JSON ---")
    print(result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()