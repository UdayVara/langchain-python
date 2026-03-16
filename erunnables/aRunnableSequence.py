from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(template="Give me a joke about {topic}", input_variables=["topic"])

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)


while True:
    user_input = input("Enter a topic to get a joke (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    response = chain.invoke({"topic": user_input})
    print(f"Joke: {response}")
    # To visualize the chain structure, you can use the `get_graph()` method to print an ASCII representation of the chain. This can help you understand how the different components are connected and how data flows through the chain.
    chain.get_graph().print_ascii()