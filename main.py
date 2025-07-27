from rag.rag_pipeline import RAGPipeline

def rag_pipeline(doc_link, questions):
    print("=== RAG Pipeline Batch Mode ===")
    rag = RAGPipeline()

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
            clear_existing=True
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
    answers = []
    for q in questions:
        answer, references = rag.answer_query(q)

        answers.append(answer)

    return answers
