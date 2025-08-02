from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import requests
from rag.rag_pipeline import RAGPipeline
from rag.vector_store import FaissVectorStore

# App setup
app = FastAPI()
bearer_scheme = HTTPBearer()

# Constants
API_TOKEN = "43fb177f633736d5eb2e45b55db7a6f647adf7614fa868c33e8a8f4eb59b4870"
EMBEDDING_DIM = 768

# Vector store instance
vector_store = FaissVectorStore(dim=EMBEDDING_DIM)

# RAG pipeline instance (shared for reuse)
rag_pipeline = RAGPipeline(embedding_dim=EMBEDDING_DIM, vector_store=vector_store)

# Security
def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Request/response schemas
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# HackRx run route
@app.post("/hackrx/run", response_model=RunResponse)
def run(payload: RunRequest, _: None = Depends(verify_token)):
    try:
        # Ingest and answer
        rag_pipeline.ingest_document(link=payload.documents)
        answers = [rag_pipeline.answer_query(q)[0] for q in payload.questions]
        print("ANSWERS:", answers)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
