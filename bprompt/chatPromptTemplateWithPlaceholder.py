from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

previousMessages = [
        ("system", "You are a helpful assistant that provides concise and accurate answers to user queries."),
        ("human", "my wifi password is 77ll993kk"),
    ]


messages = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that provides concise and accurate answers to user queries."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

print("Welcome to the chat application with Chat Prompt Template and Placeholder! Type 'exit' to quit.")
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    formatted_messages = messages.format_messages(
    history=previousMessages,
    input=user_input
    )

    response = model.invoke(formatted_messages)
    print(f"AI: {response.content}")
    messages.append(("human", user_input))
    messages.append(("assistant", response.content))