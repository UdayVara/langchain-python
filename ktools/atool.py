from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


# ✅ Define tool using decorator (recommended)
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b
