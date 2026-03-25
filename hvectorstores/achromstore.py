from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Step 1: Create Documents
# -----------------------------
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# -----------------------------
# Step 2: Load Embedding Model
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Step 3: Create Chroma DB
# -----------------------------
db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_db"
)

# -----------------------------
# Step 4: CLI Loop for Search
# -----------------------------
print("\n🔍 IPL Player Similarity Search (type 'exit' to quit)\n")

while True:
    query = input("Enter your query: ")

    if query.lower() == "exit":
        break

    results = db.similarity_search(query, k=2)

    print("\nTop Results:\n")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.page_content}")
        print(f"   👉 Team: {res.metadata['team']}\n")

    print("-" * 50)