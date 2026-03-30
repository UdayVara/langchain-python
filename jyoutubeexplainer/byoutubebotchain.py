from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ------------------ LLM ------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# ------------------ LOAD DATA ------------------
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=Rni7Fz7208c",
    add_video_info=False,
)

documents = loader.load()

# ------------------ SPLIT ------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

texts = text_splitter.split_documents(documents)

# ------------------ EMBEDDING ------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ VECTOR STORE ------------------
vectorStore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="youtube_explainer"
)

vectorStore.add_documents(texts)

retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ------------------ PROMPT ------------------
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a youtube video summarizer.

Answer the question based only on the provided context.
If you don't know, say "I don't know".

Context:
{context}

Question:
{query}
"""
)

# ------------------ HELPER ------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------ RUNNABLE CHAIN ------------------
chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "query": RunnablePassthrough(),
    }
    | prompt
    | model
)

# ------------------ LOOP ------------------
while True:
    query = input("Enter Query: ")

    if query.lower() == "exit":
        break

    response = chain.invoke(query)
    print("Answer:", response.content)