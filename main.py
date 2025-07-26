import sys
from rag.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    print("=== RAG Pipeline CLI ===")
    doc_link = input("Enter the document link (local path or URL): ").strip()
    query = input("Enter your query: ").strip()

    rag = RAGPipeline()
    
    # Check for existing data
    if rag.check_existing_data():
        print("ound existing data in vector store. This will be cleared for the new document.")
        print("   (This prevents conflicts between different documents)")
    
    print("\nIngesting document with optimizations...")
    
    try:
        rag.ingest_document(
            doc_link,
            chunk_strategy='sentences',
            target_chunks=50,
            batch_size=50,
            max_workers=4,
            clear_existing=True  # Always clear existing data for new document
        )
        
        # Check ingestion status
        status = rag.get_ingestion_status()
        print(f"\nIngestion Status:")
        print(f"Ready for queries: {status['ready_for_queries']}")
        print(f"Chunks processed: {status['chunk_count']}")
        print(f"  Vectors in database: {status['vector_count']}")
        
        if not status['ready_for_queries']:
            print("Document not ready for queries. Please check the ingestion process.")
            exit(1)
            
    except Exception as e:
        print(f"Document ingestion failed: {e}")
        exit(1)

    print("\nAnswering query...")
    answer, references = rag.answer_query(query)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== References ===\n")
    for idx, ref in references:
        print(f"[Chunk {idx}]: {ref[:200]}...\n") 