from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(template="Give me Detailed Explanation of the following topic: {topic}, in simple text without any formattings. ", input_variables=["topic"])


prompt2 = PromptTemplate(template="Give me Important Questions about the following topic: {topic}, in simple text without any formattings. ", input_variables=["topic"])


strParser = StrOutputParser()


parellelChain = RunnableParallel({
    "explanation": prompt1 | model | strParser,
    "questions": prompt2 | model | strParser
})


difficulyPrompt = PromptTemplate(template="Based on follwing topic explanation: {explanation}, & important questions: {questions}, give me difficulaty level of important questions in simple text without any formattings from low to high. ", input_variables=["explanation", "questions"])


mergedChain = parellelChain | difficulyPrompt | model | strParser

while True:
    user_input = input("Enter a topic to get a detailed explanation and important questions (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the application.")
        break
    
    response = mergedChain.invoke({"topic": user_input})
    print(f"AI Explanation: {response}")
    # To visualize the chain structure, you can use the `get_graph()` method to print an ASCII representation of the chain. This can help you understand how the different components are connected and how data flows through the chain.
    mergedChain.get_graph().print_ascii()