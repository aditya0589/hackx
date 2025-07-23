import numpy as np
from .embedder import embed_text_chunks

def retrieve_relevant_chunks(query, vector_store, all_chunks, model_name='models/embedding-001', top_k=5):
    """
    Embeds the query using Gemini embedding model, searches the FAISS vector store, and returns a list of (index, chunk) tuples for valid references.
    """
    query_embedding = embed_text_chunks([query], model_name=model_name)[0]
    query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    indices = vector_store.index.search(query_embedding, top_k)[1][0]
    # Return (index, chunk) for valid indices
    valid_refs = [(i, all_chunks[i]) for i in indices if 0 <= i < len(all_chunks)]
    return valid_refs 