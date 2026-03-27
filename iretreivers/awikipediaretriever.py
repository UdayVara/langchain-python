from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()


retriever = WikipediaRetriever()


while True:
    query = input("Enter Query to search from wikipedia: ")

    doc = retriever.invoke(query)

    print(f"Document content: {doc[0].page_content}") #doc[0].page_content)

