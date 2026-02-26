from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from amodels import index 
from bprompt import dyanmic_prompt

class llmPayload(BaseModel):
    prompt: str

class dynamicPromptPayload(BaseModel):
    paper: str
    explanationStyle: str
    explanationLength: str
    

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": test.sayHello()}

@app.post("/llm-chat")
async def ChatLLm(body: llmPayload):
    return {"response": index.generate_response(body.prompt)}


# PROMPT TEMPLATE TUTORIAL

@app.post("/dynamic-prompt")
async def DynamicPrompt(body: dynamicPromptPayload):
    response = dyanmic_prompt.generate_response(body.paper, body.explanationStyle, body.explanationLength)
    return {"response": response.content}