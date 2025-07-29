import os
import asyncio
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dotenv import load_dotenv
import time

# Explicitly load .env from the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
print(f"[DEBUG] GOOGLE_API_KEY loaded: {str(GOOGLE_API_KEY)[:5]}..." if GOOGLE_API_KEY else "[DEBUG] GOOGLE_API_KEY not found!")

import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)

def embed_text_chunks(chunks, model_name='models/embedding-001', batch_size=50, max_workers=4, max_chunk_bytes=30000):
    """
    Returns a list of embedding vectors for the given text chunks using Gemini embedding model.
    Optimized with batching and parallel processing for better performance.
    
    Args:
        chunks: List of text chunks to embed
        model_name: Name of the embedding model
        batch_size: Number of chunks to process in each batch
        max_workers: Maximum number of parallel workers
        max_chunk_bytes: Maximum bytes allowed per chunk (default: 30KB for safety)
    """
    if not chunks:
        return []
    
    # Validate chunk sizes before processing
    print(f"[INFO] Validating {len(chunks)} chunks for size limits...")
    validated_chunks = _validate_chunk_sizes(chunks, max_chunk_bytes)
    
    if len(validated_chunks) != len(chunks):
        print(f"[INFO] Chunk validation complete: {len(chunks)} -> {len(validated_chunks)} chunks")
    
    print(f"[INFO] Embedding {len(validated_chunks)} chunks with batch_size={batch_size}, max_workers={max_workers}")
    
    # Optimize batch size based on number of chunks
    if len(validated_chunks) <= 10:
        # For small datasets, use smaller batches for better reliability
        batch_size = min(batch_size, 10)
        max_workers = min(max_workers, 2)
    elif len(validated_chunks) <= 50:
        # For medium datasets, use moderate batching
        batch_size = min(batch_size, 25)
        max_workers = min(max_workers, 4)
    
    # Split chunks into batches
    batches = [validated_chunks[i:i + batch_size] for i in range(0, len(validated_chunks), batch_size)]
    
    embeddings = []
    start_time = time.time()
    
    # Process batches in parallel with improved error handling
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit batch processing tasks
        future_to_batch = {
            executor.submit(_process_batch, batch, model_name): batch 
            for batch in batches
        }
        
        # Collect results as they complete
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                embeddings.extend(batch_embeddings)
                completed_batches += 1
                print(f"[INFO] Completed batch {completed_batches}/{len(batches)} ({len(batch)} chunks)")
            except Exception as e:
                print(f"[ERROR] Failed to process batch: {e}")
                # Fallback to sequential processing for failed batch
                print(f"[INFO] Retrying batch with sequential processing...")
                fallback_embeddings = _process_batch_sequential(batch, model_name)
                embeddings.extend(fallback_embeddings)
                completed_batches += 1
    
    total_time = time.time() - start_time
    print(f"[INFO] Successfully embedded {len(embeddings)} chunks in {total_time:.2f}s")
    print(f"[INFO] Average time per chunk: {total_time/len(embeddings):.3f}s")
    
    # Return both embeddings and the validated chunks for proper tracking
    return embeddings, validated_chunks

def _process_batch(chunks_batch, model_name):
    """
    Process a batch of chunks using the Google Generative AI API with improved error handling.
    """
    try:
        # For single chunks, use direct API call
        if len(chunks_batch) == 1:
            response = genai.embed_content(
                model=model_name, 
                content=chunks_batch[0], 
                task_type="retrieval_document"
            )
            return [response['embedding']]
        else:
            # For multiple chunks, process with retry logic
            return _process_batch_with_retry(chunks_batch, model_name)
    except Exception as e:
        print(f"[WARNING] Batch processing failed, falling back to sequential: {e}")
        return _process_batch_sequential(chunks_batch, model_name)

def _process_batch_with_retry(chunks_batch, model_name, max_retries=3):
    """
    Process a batch with retry logic for better reliability.
    """
    for attempt in range(max_retries):
        try:
            embeddings = []
            for chunk in chunks_batch:
                response = genai.embed_content(
                    model=model_name, 
                    content=chunk, 
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])
            return embeddings
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[WARNING] Attempt {attempt + 1} failed, retrying...")
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff

def _process_batch_sequential(chunks_batch, model_name):
    """
    Fallback method to process chunks sequentially within a batch.
    """
    embeddings = []
    for i, chunk in enumerate(chunks_batch):
        try:
            response = genai.embed_content(
                model=model_name, 
                content=chunk, 
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"[ERROR] Failed to embed chunk {i}: {e}")
            # Add a zero vector as fallback (you might want to handle this differently)
            embeddings.append([0.0] * 768)  # Assuming 768-dimensional embeddings
    return embeddings

def _validate_chunk_sizes(chunks, max_bytes=30000):
    """
    Validate and potentially split chunks that exceed the byte limit.
    
    Args:
        chunks: List of text chunks
        max_bytes: Maximum bytes allowed per chunk
    
    Returns:
        List of validated chunks (potentially split if oversized)
    """
    from .text_splitter import validate_and_split_chunks
    return validate_and_split_chunks(chunks, max_bytes)

# Legacy function for backward compatibility
def embed_text_chunks_sequential(chunks, model_name='models/embedding-001'):
    """
    Legacy sequential embedding function for backward compatibility.
    Use embed_text_chunks() for better performance.
    """
    return _process_batch_sequential(chunks, model_name) 