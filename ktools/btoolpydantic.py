from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

load_dotenv()

class MultiplyInput(BaseModel):
    a:int = Field(description="First number")
    b:int = Field(description="Second number")

def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b


multiply = StructuredTool.from_function(
    name="multiply",
    description="Multiplies two numbers",
    func=multiply,
    input_schema=MultiplyInput,
    output_schema=int,
)


result = multiply.invoke({"a": 2, "b": 3})

print(result)