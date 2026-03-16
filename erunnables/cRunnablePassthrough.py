from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

strParser = StrOutputParser()


jokePrompt = PromptTemplate(template="Give me a joke about {topic}", input_variables=["topic"])

summarizePrompt = PromptTemplate(template="Summarize the following joke: {joke}", input_variables=["joke"])


jokeChain = RunnableSequence(jokePrompt , model , strParser)

summarizeChain = RunnableSequence(summarizePrompt , model , strParser)

passthroghParellelChain = RunnableParallel(
    {"joke": RunnablePassthrough(),
    "summary": summarizeChain}
)

finalChain = RunnableSequence(jokeChain,passthroghParellelChain)
while True:
    user_input = input("Enter a topic to get a joke (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    response = finalChain.invoke({"topic": user_input})
    print(f"Joke: {response['joke']}")
    print(f"Summary: {response['summary']}")
    print(f"Response Object: {response}")

    # To visualize the chain structure, you can use the `get_graph()` method to print an ASCII representation of the chain. This can help you understand how the different components are connected and how data flows through the chain.
    passthroghParellelChain.get_graph().print_ascii()

