import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#---------------------------------------------------------------------------------------------------------------------
# Basic LCEL example:
# input -> prompt -> model -> output parser -> text
# without structured output (Pydantic)
#---------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------
# build chain function: prompt → model → parser
#---------------------------------------------------------------------------------------------------------------------

def build_chain():

    # load env. file
    load_dotenv()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env)

    # Create a model, use: gpt-4o-mini, set creativity: 0.2
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # prompt contain 2 messages, system (general instruction), human (direct question)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please answer concisely."),
            ("human", "Explain in 3 points: {topic}"),
        ]
    )

    # LCEL: prompt → model → parser
    return prompt | model | StrOutputParser()


def main() -> None:

    # build chain and save in chain variable
    chain = build_chain()

    # run chain
    # replace {topic} with the following text: what is RAG + an example of its use.
    # save the answer to result.
    result = chain.invoke({"topic": "what is RAG + an example of its use"})
    print(result)

main()