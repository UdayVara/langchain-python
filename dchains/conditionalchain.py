from functools import partial
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

strParser = StrOutputParser()

class ResponseSchema(BaseModel):
    sentiment:Literal["positive", "negative"] = Field(..., description="The sentiment of the feedback, either 'positive' or 'negative'")

pydanticParser = PydanticOutputParser(pydantic_object=ResponseSchema)

feedbackPrompt = PromptTemplate(template="Based on the following feedback, determine if the response is positive or negative. Feedback: {feedback} \n{format_instructions} ", input_variables=["feedback"],partial_variables={"format_instructions": pydanticParser.get_format_instructions()})

classificationChain = feedbackPrompt | model | pydanticParser

positiveFeedbackPrompt = PromptTemplate(template="Based on the following positive feedback, reply with a positive response. Feedback: {feedback} ", input_variables=["feedback"])


negativeFeedbackPrompt = PromptTemplate(template="Based on the following negative feedback, reply with a constructive response. Feedback: {feedback} ", input_variables=["feedback"])

conditionalChain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', positiveFeedbackPrompt | model | strParser),
    (lambda x:x.sentiment == 'negative', negativeFeedbackPrompt | model | strParser),
    RunnableLambda(lambda x: "could not find sentiment")
)

finalChain = classificationChain | conditionalChain

while True:
    user_input = input("Enter feedback to get a response (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    response = finalChain.invoke({"feedback": user_input})
    print(f"AI Response: {response}")
    # To visualize the chain structure, you can use the `get_graph()` method to print an ASCII representation of the chain. This can help you understand how the different components are connected and how data flows through the chain.
    finalChain.get_graph().print_ascii()