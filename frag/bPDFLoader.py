from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


pdfLoader = PyPDFLoader("frag/bPDFTest.pdf",)

documents = pdfLoader.load()

pdfData = "\n".join([doc.page_content for doc in documents])
prompt = PromptTemplate(template="Answer the following question based on the provided document: {document} \n Question: {question}", input_variables=["document", "question"])

strParser = StrOutputParser()

chain = prompt | model | strParser

while True:
    user_input = input("Enter a question to get an answer based on the document (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    response = chain.invoke({"document": pdfData, "question": user_input})
    print(f"Response: {response}")

