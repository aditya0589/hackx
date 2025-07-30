from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
from rag.text_splitter import split_text, validate_and_split_chunks
from rag.vector_store import FaissVectorStore

app = FastAPI()

# Create an API router with prefix /api/v1
api_v1 = APIRouter(prefix="/api/v1")

# Initialize vector store on app start
DIM = 1536  # Embedding vector dimension
INDEX_PATH = 'faiss.index'
META_PATH = 'faiss_meta.pkl'
vector_store = FaissVectorStore(dim=DIM, index_path=INDEX_PATH, meta_path=META_PATH)

# Dummy embedding function - replace with your real embedding code
def embed_text(text):
    import numpy as np
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(DIM).astype('float32').tolist()

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

    # Return matched texts and optionally metadata
    return {"results": results}

@api_v1.get("/info")
async def info():
    stats = vector_store.meta and len(vector_store.meta) or 0
    return {"total_vectors": stats}

# Include the router in the main app
app.include_router(api_v1)