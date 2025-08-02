import numpy as np
from .embedder import embed_text_chunks

def retrieve_relevant_chunks(query, vector_store, all_chunks, model_name='models/embedding-001', top_k=5, context_ratio=0.3):
    """
    Retrieves chunks using a hybrid approach:
    1. Gets the most relevant chunks based on semantic similarity
    2. Includes additional chunks for better context
    
    Args:
        query: The user's query
        vector_store: The FAISS vector store
        all_chunks: List of all document chunks
        model_name: Name of the embedding model
        top_k: Number of most relevant chunks to retrieve
        context_ratio: Ratio of additional context chunks to include (0.0 to 1.0)
    
    Returns:
        List of (index, chunk) tuples with relevant and context chunks
    """
    # Get query embedding
    query_result = embed_text_chunks([query], model_name=model_name)
    
    # Handle the new return format (embeddings, validated_chunks)
    if isinstance(query_result, tuple):
        query_embeddings, _ = query_result
    else:
        query_embeddings = query_result
    
    query_embedding = query_embeddings[0]
    
    # Get the most relevant chunks based on semantic similarity
    relevant_chunks = vector_store.search(query_embedding, top_k=top_k)
    
    # Get indices of relevant chunks
    relevant_indices = set()
    for chunk in relevant_chunks:
        try:
            idx = all_chunks.index(chunk)
            relevant_indices.add(idx)
        except ValueError:
            # If chunk not found, skip
            continue
    
    # Calculate how many additional context chunks to include
    total_chunks = len(all_chunks)
    context_chunks_count = int(total_chunks * context_ratio)
    
    # Get additional context chunks (excluding already selected relevant chunks)
    context_indices = set()
    remaining_indices = [i for i in range(total_chunks) if i not in relevant_indices]
    
    # Include chunks around the relevant chunks for better context
    for idx in relevant_indices:
        # Add chunks before and after relevant chunks
        for offset in range(-2, 3):  # -2, -1, 0, 1, 2
            context_idx = idx + offset
            if 0 <= context_idx < total_chunks and context_idx not in relevant_indices:
                context_indices.add(context_idx)
    
    # If we still need more context chunks, add some from the remaining chunks
    if len(context_indices) < context_chunks_count:
        additional_needed = context_chunks_count - len(context_indices)
        # Add chunks from the beginning, middle, and end of the document
        for i in range(0, min(additional_needed // 3, len(remaining_indices))):
            if remaining_indices[i] not in context_indices:
                context_indices.add(remaining_indices[i])
        
        # Add from middle
        mid_start = len(remaining_indices) // 2
        for i in range(mid_start, min(mid_start + additional_needed // 3, len(remaining_indices))):
            if remaining_indices[i] not in context_indices:
                context_indices.add(remaining_indices[i])
        
        # Add from end
        end_start = max(0, len(remaining_indices) - additional_needed // 3)
        for i in range(end_start, len(remaining_indices)):
            if remaining_indices[i] not in context_indices:
                context_indices.add(remaining_indices[i])
    
    # Combine relevant and context indices
    all_selected_indices = list(relevant_indices | context_indices)
    all_selected_indices.sort()  # Sort to maintain document order
    
    # Return chunks with their indices
    result = [(idx, all_chunks[idx]) for idx in all_selected_indices]
    
    return result 