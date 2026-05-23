# pip install langchain langchain-openai python-dotenv pydantic

import os
from dotenv import load_dotenv #allows you to load an .env file
from pydantic import BaseModel, Field # BaseModel = the base for creating the schema / Field = the field description

from langchain_openai import ChatOpenAI # class to connect to OpenAI model via LangChain
from langchain_core.prompts import ChatPromptTemplate # to make the prompt organized
from langchain_core.output_parsers import StrOutputParser # Model response -> plain text

#---------------------------------------------------------------------------------------------------------------------
# Basic LCEL with structured output (Pydantic):
# input -> prompt -> model -> output parser -> text
# input text → prompt → model with schema → object Pydantic
#---------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------
# I want to get an object that has exactly these fields: topic, action, deadline, priority
# I crated my own data type - Extracted
#---------------------------------------------------------------------------------------------------------------------
class Extracted(BaseModel):
    topic: str = Field(description="Krótki temat tekstu")
    action: str = Field(description="Co trzeba zrobić (jedno zdanie)")
    deadline: str | None = Field(description="Termin jeśli jest w tekście, inaczej null")
    priority: str = Field(description="Jedno z: low, medium, high")


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Wyciągaj dane i zwracaj je zgodnie ze schematem."),
            ("human", "Tekst:\n{text}\n\nWyciągnij pola ze schematu."),
        ]
    )

    # A) LCEL -> wynik jako TEKST
    text_chain = prompt | model | StrOutputParser()

    # B) LCEL -> wynik jako OBIEKT (structured output)
    structured_model = model.with_structured_output(Extracted)
    structured_chain = prompt | structured_model

    sample = "Hej, potrzebuję do jutra do 18:00 przygotować krótkie podsumowanie o RAG. To ważne."

    print("=== TEXT OUTPUT ===")
    print(text_chain.invoke({"text": sample}))

    print("\n=== STRUCTURED OUTPUT (Pydantic object) ===")
    obj = structured_chain.invoke({"text": sample})
    print(obj)

    print("\n=== STRUCTURED OUTPUT JSON ===")
    print(obj.model_dump_json(indent=2, ensure_ascii=False))



main()