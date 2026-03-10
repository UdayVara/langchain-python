from ast import parse

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


class outputScehema(BaseModel):
    name:str = Field(description="Name of the person")
    age:int = Field(gt=18,lt=100,description="Age of the person")
    city:str = Field(description="City of residence")


parser = PydanticOutputParser(pydantic_object=outputScehema)

prompt = PromptTemplate(template="Generate the Fictional  details of person with follwing nationality: {nationality} \n{format_instructions}", input_variables=["nationality"],partial_variables={"format_instructions": parser.get_format_instructions()})





chain = prompt | model | parser


while True:
    user_input = input("Enter Nationality to get the fictional name details (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break

    # Manually invoking the chain components to show the parsing step clearly
    # formatted_prompt = prompt.invoke({"nationality": user_input})
    # response = model.invoke(formatted_prompt)
    # parsed_response = parser.parse(response.content)
    # print(f"AI: {parsed_response}")


    # Automatically invoking the chain
    response = chain.invoke({"nationality": user_input})
    print(f"AI: {response}")