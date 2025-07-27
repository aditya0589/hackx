# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# from main import rag_pipeline
# app = FastAPI()

# class RunRequest(BaseModel):
#     documents: str
#     questions: List[str]

# class RunResponse(BaseModel):
#     answers: List[str]


# @app.post("/hackrx/run", response_model=RunResponse)
# def run_endpoint(payload: RunRequest):
#     try:
#         final = rag_pipeline(payload.documents, payload.questions[0])  
#         return {"answers": [final]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# from fastapi import FastAPI, HTTPException, Header, Depends
# from pydantic import BaseModel
# from typing import List, Optional
# from main import rag_pipeline

# app = FastAPI()

# class RunRequest(BaseModel):
#     documents: str
#     questions: List[str]

# class RunResponse(BaseModel):
#     answers: List[str]

# # Simple token check (can be replaced with real auth logic)
# def verify_token(authorization: Optional[str] = Header(None)):
#     if authorization != "Bearer 43fb177f633736d5eb2e45b55db7a6f647adf7614fa868c33e8a8f4eb59b4870":
#         raise HTTPException(status_code=401, detail="Unauthorized")


# @app.post("/hackrx/run", response_model=RunResponse)
# def run_endpoint(payload: RunRequest, _: None = Depends(verify_token)):
#     try:
#         final = rag_pipeline(payload.documents, payload.questions[0])
#         return {"answers": [final]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from main import rag_pipeline

app = FastAPI()

# Security scheme for Swagger and API usage
bearer_scheme = HTTPBearer()

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

# Endpoint
@app.post("/hackrx/run", response_model=RunResponse)
def run_endpoint(payload: RunRequest, _: None = Depends(verify_token)):
    try:
        answers = rag_pipeline(payload.documents, payload.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

