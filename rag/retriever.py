import numpy as np
from .embedder import embed_text_chunks

def retrieve_relevant_chunks(query, vector_store, all_chunks, model_name='models/embedding-001', top_k=5):
    """
    Returns a list of (index, chunk) tuples for all chunks in the document, regardless of the query.
    """
    return [(i, chunk) for i, chunk in enumerate(all_chunks)] 
