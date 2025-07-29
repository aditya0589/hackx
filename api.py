from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from main import rag_pipeline
import time
import uuid
from performance_monitor import monitor

app = FastAPI()

# Security scheme for Swagger and API usage
bearer_scheme = HTTPBearer()

# Global cache for RAG pipeline instances
rag_cache = {}

# Validate token
def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials
    if token != "43fb177f633736d5eb2e45b55db7a6f647adf7614fa868c33e8a8f4eb59b4870":
        raise HTTPException(status_code=401, detail="Unauthorized")

# Request model
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# Response model
class RunResponse(BaseModel):
    answers: List[str]

def get_cached_rag_pipeline(doc_link):
    """Get or create a cached RAG pipeline for the document."""
    if doc_link in rag_cache:
        return rag_cache[doc_link]
    
    # Create new pipeline and cache it
    from rag.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    rag_cache[doc_link] = rag
    return rag

# Endpoint
@app.post("/hackrx/run", response_model=RunResponse)
def run_endpoint(payload: RunRequest, _: None = Depends(verify_token)):
    request_id = str(uuid.uuid4())[:8]
    start_metrics = monitor.start_monitoring()
    
    try:
        start_time = time.time()
        
        # Get cached or create new RAG pipeline
        rag = get_cached_rag_pipeline(payload.documents)
        
        # Check if document is already ingested
        status = rag.get_ingestion_status()
        
        if not status['ready_for_queries']:
            print(f"[INFO] Document not ready, ingesting...")
            rag.ingest_document(
                payload.documents,
                chunk_strategy='sentences',
                target_chunks=50,
                batch_size=50,
                max_workers=4,
                clear_existing=False,  # Don't clear existing data
                use_cache=True  # Use caching
            )
        
        # Answer questions
        answers = []
        for i, question in enumerate(payload.questions):
            print(f"[INFO] Processing question {i+1}/{len(payload.questions)}")
            answer, references = rag.answer_query(question)
            answers.append(answer)
        
        total_time = time.time() - start_time
        print(f"[INFO] Request {request_id} completed in {total_time:.2f}s")
        
        monitor.end_monitoring(request_id, True)
        return {"answers": answers}
    except Exception as e:
        print(f"[ERROR] Request {request_id} failed: {e}")
        monitor.end_monitoring(request_id, False, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/performance")
def performance_summary():
    """Get performance summary."""
    summary = monitor.get_performance_summary()
    return summary

