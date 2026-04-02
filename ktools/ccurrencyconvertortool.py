from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import requests

load_dotenv()

# 🔥 Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # fast + free tier
    temperature=0
)


# 🔥 Step 1: Schema
class CurrencyInput(BaseModel):
    amount: float = Field(..., description="Amount to convert")
    from_currency: str = Field(..., description="Source currency (INR, USD, etc)")
    to_currency: str = Field(..., description="Target currency")


# 🔥 Step 2: Function
def convert_currency_func(amount: float, from_currency: str, to_currency: str) -> str:
    url = "https://api.exchangerate.host/convert"

    params = {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "amount": amount
    }

    print(f"🔧 Tool Called: {amount} {from_currency.upper()} → {to_currency.upper()}")

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "result" not in data:
            return "Error fetching conversion"

        return f"{amount} {from_currency.upper()} = {round(data['result'], 2)} {to_currency.upper()}"

    except Exception as e:
        return f"Error: {str(e)}"


# 🔥 Step 3: Structured Tool
convert_currency_tool = StructuredTool.from_function(
    name="convert_currency",
    description="Use this tool for real-time currency conversion. Always use this tool for currency queries.",
    func=convert_currency_func,
    args_schema=CurrencyInput,
)


# 🔥 Bind tools (Gemini supports this ✅)
model_with_tools = llm.bind_tools([convert_currency_tool])


# 🔥 Chat loop
while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    response = model_with_tools.invoke(user_input)

    print("AI:", response.content)