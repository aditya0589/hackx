from rag.rag_pipeline import RAGPipeline
import time

# Global cache for RAG instances
rag_instances = {}

def rag_pipeline(doc_link, questions):
    print("=== RAG Pipeline Optimized Mode ===")
    
    # Check if we have a cached instance for this document
    if doc_link in rag_instances:
        rag = rag_instances[doc_link]
        print("Using cached RAG instance")
    else:
        rag = RAGPipeline()
        rag_instances[doc_link] = rag
        print("Created new RAG instance")

    # Check for existing data
    if rag.check_existing_data():
        print("Found existing data in vector store. This will be cleared for the new document.")
    
    print("\nIngesting document with optimizations...")

    try:
        rag.ingest_document(
            doc_link,
            chunk_strategy='sentences',
            target_chunks=50,
            batch_size=50,
            max_workers=4,
            clear_existing=True,
            use_cache=True  # Enable caching
        )

        # Check ingestion status
        status = rag.get_ingestion_status()
        print(f"\nIngestion Status:")
        print(f"Ready for queries: {status['ready_for_queries']}")
        print(f"Chunks processed: {status['chunk_count']}")
        print(f"Vectors in database: {status['vector_count']}")

        if not status['ready_for_queries']:
            raise Exception("Document not ready for queries.")

    except Exception as e:
        print(f"Document ingestion failed: {e}")
        raise e

    print("\nAnswering questions...")
    start_time = time.time()
    answers = []
    for i, q in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {q[:50]}...")
        answer, references = rag.answer_query(q)
        answers.append(answer)
    
    total_time = time.time() - start_time
    print(f"All questions answered in {total_time:.2f}s")

    return answers
