from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm  = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm) 

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=Rni7Fz7208c", add_video_info=False
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


texts = text_splitter.split_documents(documents)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorStore = Chroma(persist_directory="./chroma_db", embedding_function=embedding, collection_name="youtube_explainer")


vectorStore.add_documents(texts)

retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt = PromptTemplate(
    input_variables=["query","context"],
    template="You are youtube video summarizer and your task is to answer the question based on the provided context & if you don't know the answer then just say that you don't know. \n\n Context: {context}\n\n Question: {query}",
)

while True:
    query = input("Enter Query to search from chroma db: ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(query=query, context=context)
    response = model.invoke(final_prompt)
    print(f"Answer: {response.content}")    