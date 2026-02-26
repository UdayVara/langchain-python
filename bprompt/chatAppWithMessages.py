from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

last_chats = []

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    last_chats.append(HumanMessage(user_input)) 
    response = model.invoke(last_chats)
    last_chats.append(SystemMessage(response.content)) 
    print(f"AI: {response.content}")

print("Chat history:", last_chats)