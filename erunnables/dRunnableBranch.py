from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class ResponseSchema(BaseModel):
    sentiment:Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback, either 'positive' or 'negative'")

pydanticParser = PydanticOutputParser(pydantic_object=ResponseSchema)

strParser = StrOutputParser()
reviewPrompt  = PromptTemplate(template="Based on the following review, determine if the review is positive or negative. Review: {review} \n {response_format}", input_variables=["review"],partial_variables={"response_format": pydanticParser.get_format_instructions()})

positivePrompt = PromptTemplate(template="The following review is positive: {review}, reply with a positive response", input_variables=["review"])

negativePrompt = PromptTemplate(template="The following review is negative: {review}, reply with a constructive response", input_variables=["review"])

positiveChain = RunnableSequence(positivePrompt , model , strParser)
negativeChain = RunnableSequence(negativePrompt , model , strParser)

classificationChain = reviewPrompt | model | pydanticParser


distiguishChain = RunnableBranch(
    ((lambda x: x.sentiment == "positive"), positiveChain),
    ((lambda x: x.sentiment == "negative"), negativeChain),
    RunnablePassthrough()
)

finalChain = RunnableSequence(classificationChain, distiguishChain)


while True:
    user_input = input("Enter a review to get feedback (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    response = finalChain.invoke({"review": user_input})
    print(f"Response: {response}")