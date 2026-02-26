from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Document embeddings
doc_embeddings = embedding.embed_documents(documents)

# Query embedding
query = "what is capital of paris"
query_embedding = embedding.embed_query(query)

# Compute cosine similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get highest score
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)