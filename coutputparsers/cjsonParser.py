from click import prompt
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)




parser = JsonOutputParser()

prompt = PromptTemplate(template="Give me Detailed Explanation of the following topic: {topic} \n{format_instructions}", input_variables=["topic"],partial_variables={"format_instructions": parser.get_format_instructions()})

chain = prompt | model | parser

while True:
    user_input = input("Enter a topic to get a detailed explanation (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break

    
    response = chain.invoke({"topic": user_input})
    
    print(f"AI: {response}")