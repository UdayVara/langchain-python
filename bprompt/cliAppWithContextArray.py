from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

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
    
    last_chats.append({"role": "user", "content": user_input}) 
    response = model.invoke(last_chats)
    last_chats.append({"role": "assistant", "content": response.content}) 
    print(f"AI: {response.content}")   
    