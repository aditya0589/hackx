import os
from dotenv import load_dotenv

# Explicitly load .env from the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print(f"[DEBUG] GOOGLE_API_KEY loaded: {str(GOOGLE_API_KEY)[:5]}..." if GOOGLE_API_KEY else "[DEBUG] GOOGLE_API_KEY not found!")

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

# Embed a list of text chunks using Gemini embedding model (not generation model)
def embed_text_chunks(chunks, model_name='models/embedding-001'):
    """
    Returns a list of embedding vectors for the given text chunks using Gemini embedding model.
    """
    embeddings = []
    for chunk in chunks:
        response = genai.embed_content(model=model_name, content=chunk, task_type="retrieval_document")
        embeddings.append(response['embedding'])
    return embeddings 