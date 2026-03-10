
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(template="Give me Detailed Explanation of the following topic: {topic}, in simple text without any formattings. ", input_variables=["topic"])

summarisePrompt = PromptTemplate(template="Summarise the following topic: {text} in less than 50 words", input_variables=["text"])

while True:
    user_input = input("Enter a topic to get a detailed explanation (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    formatted_prompt = prompt.invoke({"topic": user_input})
    response_1 = model.invoke(formatted_prompt)

    formatted_summarisePrompt = summarisePrompt.invoke({"text": response_1.content})
    response_2 = model.invoke(formatted_summarisePrompt)
    print(f"AI: {response_2.content}")


