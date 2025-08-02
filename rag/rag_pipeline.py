# rag_pipeline.py
from .document_loader import load_document
from .text_splitter import split_text, optimize_chunk_size, get_chunk_size_info
from .embedder import embed_text_chunks
from .vector_store import FaissVectorStore
from .retriever import retrieve_relevant_chunks
import google.generativeai as genai
import time

class RAGPipeline:
    def __init__(self, embedding_dim=768,  vector_store=None):
        self.embedding_dim = embedding_dim
        self.vector_store = vector_store or FaissVectorStore(dim=embedding_dim)
        self.chunks = []
        self.embedding_model = 'models/embedding-001'
        self.generation_model = 'models/gemini-1.5-flash'
        self.is_ingested = False

    def ingest_document(self, link, chunk_strategy='sentences', target_chunks=50, batch_size=50, max_workers=4, clear_existing=True):
        start_time = time.time()

        if clear_existing:
            print("[INFO] Clearing existing vector store...")
            self.vector_store.clear()
            self.is_ingested = False

        print("[INFO] Loading document...")
        text, _ = load_document(link)
        if not text or not text.strip():
            raise ValueError("The document contains no extractable text.")

        print("[INFO] Splitting text into chunks...")
        optimal_chunk_size = optimize_chunk_size(len(text), target_chunks)
        self.chunks = split_text(
            text,
            chunk_size=optimal_chunk_size,
            overlap=max(100, optimal_chunk_size // 5),
            strategy=chunk_strategy
        )

        if not self.chunks:
            raise ValueError("The document could not be split into chunks.")

        size_info = get_chunk_size_info(self.chunks)
        print(f"[INFO] Total Chunks: {size_info['total_chunks']}, Avg Size: {size_info['avg_bytes']} bytes")

        print("[INFO] Generating embeddings...")
        result = embed_text_chunks(
            self.chunks,
            model_name=self.embedding_model,
            batch_size=batch_size,
            max_workers=max_workers,
            max_chunk_bytes=30000
        )

        if isinstance(result, tuple):
            embeddings, validated_chunks = result
        else:
            embeddings = result
            validated_chunks = self.chunks

        if not embeddings:
            raise ValueError("No embeddings could be generated.")

        print("[INFO] Storing in vector database...")
        self.vector_store.add(embeddings, validated_chunks)
        self.chunks = validated_chunks

        self._verify_ingestion()
        self.is_ingested = True
        print("[INFO] Document fully loaded and ready for queries")

    def answer_query(self, query, top_k=5, context_ratio=0.3):
        if not self.is_ingested:
            raise ValueError("Document has not been fully ingested yet.")

        references = retrieve_relevant_chunks(
            query,
            self.vector_store,
            self.chunks,
            model_name=self.embedding_model,
            top_k=top_k,
            context_ratio=context_ratio
        )

        context = '\n'.join(chunk for _, chunk in references)
        prompt = f"""
You are an expert assistant. Based on the context, answer the question concisely.
If the provided context is insufficient, return the best assumption based on the nearest relevant data.
Context:
{context}

Question: {query}
Answer:
"""
        model = genai.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip(), references

    def _verify_ingestion(self):
        actual_count = self.vector_store.index.ntotal
        if actual_count == 0:
            raise ValueError("Vector store is empty.")

        final_chunk_count = len(self.vector_store.meta) if hasattr(self.vector_store, 'meta') else actual_count
        print(f"[INFO] Verification: {actual_count} vectors stored for {final_chunk_count} chunks")

        try:
            test_query = "test"
            result = embed_text_chunks([test_query], model_name=self.embedding_model)
            if isinstance(result, tuple):
                test_embeddings, _ = result
            else:
                test_embeddings = result
            test_embedding = test_embeddings[0]
            test_results = self.vector_store.search(test_embedding, top_k=1)
            if not test_results:
                raise ValueError("Retrieval test failed.")
        except Exception as e:
            raise ValueError(f"Verification failed: {e}")
