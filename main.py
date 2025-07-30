# main.py
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from rag.text_splitter import split_text, validate_and_split_chunks
from rag.vector_store import FaissVectorStore
from rag.rag_pipeline import RAGPipeline

app = FastAPI()

# ---------- CONFIGURATION ----------
DIM = 1536
INDEX_PATH = 'faiss.index'
META_PATH = 'faiss_meta.pkl'
vector_store = FaissVectorStore(dim=DIM, index_path=INDEX_PATH, meta_path=META_PATH)

# ---------- EMBEDDING FUNCTION ----------
def embed_text(text):
    import numpy as np
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(DIM).astype('float32').tolist()

# ---------- SECURITY ----------
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials
    if token != "43fb177f633736d5eb2e45b55db7a6f647adf7614fa868c33e8a8f4eb59b4870":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- /api/v1 ROUTES ----------
api_v1 = APIRouter(prefix="/api/v1")

class IngestRequest(BaseModel):
    text: str
    chunk_size: int = 1000
    overlap: int = 200
    strategy: str = 'words'

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@api_v1.post("/ingest")
async def ingest(req: IngestRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    chunks = split_text(req.text, chunk_size=req.chunk_size, overlap=req.overlap, strategy=req.strategy)
    chunks = validate_and_split_chunks(chunks)

    embeddings = [embed_text(chunk) for chunk in chunks]
    metadatas = [{'text': chunk} for chunk in chunks]

    vector_store.add(embeddings, metadatas)
    return {"message": f"Ingested {len(chunks)} chunks successfully"}

@api_v1.post("/query")
async def query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query provided")

    query_emb = embed_text(req.query)
    results = vector_store.search(query_emb, top_k=req.top_k)
    return {"results": results}

@api_v1.get("/info")
async def info():
    stats = vector_store.meta and len(vector_store.meta) or 0
    return {"total_vectors": stats}

app.include_router(api_v1)

# ---------- /hackrx/run ROUTE ----------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]



def rag_pipeline(doc_url: str, questions: List[str]) -> List[str]:
    try:
        response = requests.get(doc_url)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download document: {e}")

    pdf_reader = PdfReader(BytesIO(response.content))
    full_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    if not full_text.strip():
        raise ValueError("The downloaded PDF is empty or unreadable")

    chunks = split_text(full_text, chunk_size=1000, overlap=200, strategy="words")
    chunks = validate_and_split_chunks(chunks)

    embeddings = [embed_text(chunk) for chunk in chunks]
    metadatas = [{'text': chunk} for chunk in chunks]

    vector_store.clear()
    vector_store.add(embeddings, metadatas)

    answers = []
    for q in questions:
        q_emb = embed_text(q)
        ragp= RAGPipeline()
        answers.append(ragp.answer_query(query=q_emb))

    return answers

@app.post("/hackrx/run", response_model=RunResponse)
def run_endpoint(payload: RunRequest, _: None = Depends(verify_token)):
    try:
        answers = rag_pipeline(payload.documents, payload.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
