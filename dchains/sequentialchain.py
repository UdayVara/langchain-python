from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


strParser = StrOutputParser()

prompt = PromptTemplate(template="Give me Detailed Explanation of the following topic: {topic}, in simple text without any formattings. ", input_variables=["topic"])

summarisePrompt = PromptTemplate(template="Summarise the following topic: {text} in 5 point summary", input_variables=["text"])

chain = prompt | model | strParser | summarisePrompt | model | strParser

while True:
    user_input = input("Enter a topic to get a detailed explanation (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    response = chain.invoke({"topic": user_input})
    print(f"AI: {response}")
