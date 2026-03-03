from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


messages = ChatPromptTemplate([
    ("system", "You are a helpful assistant that provides concise and accurate answers to user queries."),
    ("human", "{input}"),
])

print("Welcome to the chat application with Chat Prompt Template! Type 'exit' to quit.")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    prompt = messages.invoke({"input": user_input})
    response = model.invoke(prompt)
    print(f"AI: {response.content}")
    messages.append(("human", user_input))
    messages.append(("assistant", response.content))

