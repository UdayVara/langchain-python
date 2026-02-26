from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ThomasJohn/qlora2",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

def generate_response(prompt):
    response = model.invoke(prompt)
    return response