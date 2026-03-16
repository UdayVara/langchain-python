from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

whastapp_prompt = PromptTemplate(template="Give me Whastapp message to apply for this job {topic}", input_variables=["topic"])

email_prompt = PromptTemplate(template="Give me email to apply for this job {topic}", input_variables=["topic"])

job_resposne_chain = RunnableParallel(
    {"whatsapp": RunnableSequence(whastapp_prompt, model, parser),
     "email": RunnableSequence(email_prompt, model, parser )}
)


while True:
    user_input = input("Enter a job title to get application messages (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    response = job_resposne_chain.invoke({"topic": user_input})
    print(f"Whastapp Message: {response['whatsapp']}")
    print(f"Email: {response['email']}")
    print("Raw Response:", response)