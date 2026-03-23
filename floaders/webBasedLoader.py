from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


webLoader = WebBaseLoader(web_path="https://en.wikipedia.org/wiki/2026_Men%27s_T20_World_Cup_squads")


documents = webLoader.load()


prompt = PromptTemplate(template="Answer the following question based on the provided document: {document} \n Question: {question}", input_variables=["document", "question"])

print(f"Document content: {documents[0].page_content}")
while True:
    user_input = input("Enter a question to get an answer based on the document (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    responsePrompt = prompt.invoke({"document": documents[0].page_content, "question": user_input})

    response = model.invoke(responsePrompt)

    print(f"Response: {response}")