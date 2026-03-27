
from math import e

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(persist_directory="./chroma_db",embedding_function=embedding)

retriever = db.as_retriever()


while True:
    query = input("Enter Query to search from chroma db: ")

    if query.lower() == "exit":
        break

    doc = retriever.invoke(query)
    # print(f"DOC: {doc}")
    print(f"Document content: {doc[0].page_content}") #doc[0].page_content)