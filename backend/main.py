from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import get_agent
import re
from contextlib import asynccontextmanager
from agents.factory import initialize_multi_agent_system
import os 

API_KEY = os.getenv("OPENAI_API_KEY") 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def clean_output(output):
    cleaned_output = re.sub(r'\s+', ' ', output)  
    cleaned_output = re.sub(r'[^\x00-\x7F]+', '', cleaned_output) 
    return cleaned_output.strip()

def extract_output_from_string(data):
    match = re.search(r"'output':\s*\"(.*?)\"", data, re.DOTALL)
    if match:
        return match.group(1)
    return ''


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing AI system...")

    app.state.supervisor = initialize_multi_agent_system(
        directory_path="data_ingress/mom/webpage",
        api_key=API_KEY,
    )

    print("AI system ready")

    yield

    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):

    supervisor = app.state.supervisor

    answer = supervisor.run(req.message)

    return {
        "response": answer
    }
