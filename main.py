import sys
from rag.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    print("=== RAG Pipeline CLI ===")
    doc_link = input("Enter the document link (local path or URL): ").strip()
    query = input("Enter your query: ").strip()

    rag = RAGPipeline()
    print("\nIngesting document...")
    rag.ingest_document(doc_link)
    print("Document ingested.\n")

    print("Answering query...")
    answer, references = rag.answer_query(query)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== References ===\n")
    for idx, ref in references:
        print(f"[Chunk {idx}]: {ref[:200]}...\n") 