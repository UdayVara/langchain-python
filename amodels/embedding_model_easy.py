from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# 1. Create a vector store from your texts
vectorstore = DocArrayInMemorySearch.from_texts(documents, embedding)

# 2. Search directly
query = "what is capital of paris"
# k=1 ensures we only get the single best match
results = vectorstore.similarity_search_with_score(query, k=1)

# 3. Extract results
doc, score = results[0]

print(f"Query: {query}")
print(f"Most similar document: {doc.page_content}")
print(f"Similarity score (Distance): {score}")