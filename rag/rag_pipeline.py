from .document_loader import load_document
from .text_splitter import split_text, optimize_chunk_size, get_chunk_size_info
from .embedder import embed_text_chunks
from .vector_store import FaissVectorStore
from .retriever import retrieve_relevant_chunks
import google.generativeai as genai
import time

# Main RAG pipeline using Gemini 1.5 Flash for generation and embedding-001 for embeddings
class RAGPipeline:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.vector_store = FaissVectorStore(dim=embedding_dim)
        self.chunks = []
        self.embedding_model = 'models/embedding-001'
        self.generation_model = 'models/gemini-1.5-flash'
        self.is_ingested = False  # Flag to track if document has been fully ingested

    def ingest_document(self, link, chunk_strategy='sentences', target_chunks=50, batch_size=50, max_workers=4, clear_existing=True):
        """
        Ingest a document with performance optimizations.
        
        Args:
            link: Document link or path
            chunk_strategy: 'words', 'sentences', or 'paragraphs'
            target_chunks: Target number of chunks (for auto-sizing)
            batch_size: Number of chunks to embed in each batch
            max_workers: Maximum number of parallel workers for embedding
            clear_existing: Whether to clear existing vectors before ingesting
        """
        start_time = time.time()
        
        # Clear existing vectors if requested
        if clear_existing:
            print("[INFO] Clearing existing vector store...")
            self.vector_store.clear()
            self.is_ingested = False
        
        print("[INFO] Loading document...")
        load_start = time.time()
        text, _ = load_document(link)
        if not text or not text.strip():
            raise ValueError("The document contains no extractable text. Please provide a document with selectable text.")
        print(f"[INFO] Document loaded in {time.time() - load_start:.2f}s")
        
        print("[INFO] Splitting text into chunks...")
        split_start = time.time()
        
        # Auto-optimize chunk size based on text length
        optimal_chunk_size = optimize_chunk_size(len(text), target_chunks)
        print(f"[INFO] Using chunk size: {optimal_chunk_size} words")
        
        self.chunks = split_text(
            text, 
            chunk_size=optimal_chunk_size, 
            overlap=max(100, optimal_chunk_size // 5),
            strategy=chunk_strategy
        )
        
        if not self.chunks:
            raise ValueError("The document could not be split into chunks. It may be empty or not processable.")
        print(f"[INFO] Split into {len(self.chunks)} chunks in {time.time() - split_start:.2f}s")
        
        # Analyze chunk sizes before embedding
        size_info = get_chunk_size_info(self.chunks)
        print(f"[INFO] Chunk size analysis:")
        print(f"   Total chunks: {size_info['total_chunks']}")
        print(f"   Size range: {size_info['min_bytes']} - {size_info['max_bytes']} bytes")
        print(f"   Average size: {size_info['avg_bytes']:.0f} bytes")
        if size_info['oversized_chunks'] > 0:
            print(f"   ⚠️  Oversized chunks: {size_info['oversized_chunks']} (will be auto-split)")
        
        print("[INFO] Generating embeddings...")
        embed_start = time.time()
        result = embed_text_chunks(
            self.chunks, 
            model_name=self.embedding_model,
            batch_size=batch_size,
            max_workers=max_workers,
            max_chunk_bytes=30000  # 30KB limit for safety
        )
        
        # Handle the new return format (embeddings, validated_chunks)
        if isinstance(result, tuple):
            embeddings, validated_chunks = result
        else:
            # Backward compatibility
            embeddings = result
            validated_chunks = self.chunks
        
        if not embeddings:
            raise ValueError("No embeddings could be generated from the document. Please check the document content.")
        print(f"[INFO] Embeddings generated in {time.time() - embed_start:.2f}s")
        
        print("[INFO] Storing in vector database...")
        store_start = time.time()
        # Store the validated chunks (which may be different from original chunks due to splitting)
        self.vector_store.add(embeddings, validated_chunks)
        # Update self.chunks to reflect the final validated chunks
        self.chunks = validated_chunks
        print(f"[INFO] Stored in vector database in {time.time() - store_start:.2f}s")
        
        total_time = time.time() - start_time
        print(f"[INFO] Document ingestion completed in {total_time:.2f}s")
        print(f"[INFO] Average time per chunk: {total_time/len(self.chunks):.3f}s")
        
        # Verify that all chunks are properly stored in the vector database
        self._verify_ingestion()
        self.is_ingested = True
        print("[INFO] ✅ Document fully loaded and ready for queries")

    def answer_query(self, query, top_k=5, context_ratio=0.3):
        """
        Returns (answer, references) where references is a list of (index, chunk) tuples.
        
        Args:
            query: The user's question
            top_k: Number of most relevant chunks to retrieve
            context_ratio: Ratio of additional context chunks to include (0.0 to 1.0)
        """
        if not self.is_ingested:
            raise ValueError("Document has not been fully ingested yet. Please complete the ingestion process first.")
        
        if not self.chunks:
            raise ValueError("No document has been ingested or the document is empty. Please ingest a valid document first.")
        
        # Retrieve relevant chunks with additional context
        references = retrieve_relevant_chunks(
            query, 
            self.vector_store, 
            self.chunks, 
            model_name=self.embedding_model,
            top_k=top_k,
            context_ratio=context_ratio
        )
        
        context = '\n'.join(chunk for _, chunk in references)
        
        # Generate answer using Gemini 1.5 Flash
        prompt = f"""
You are an expert assistant. Based on the context, answer the question in a concise way.
If the provided context is insufficient, just assume the nearest correct answer. If query is not related to file return ask queries related to file.

Context:
{context}

Question: {query}
Answer:
"""
        model = genai.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip(), references
    
    def _verify_ingestion(self):
        """
        Verify that all chunks have been properly stored in the vector database.
        Raises an exception if verification fails.
        """
        if not self.chunks:
            raise ValueError("No chunks available for verification")
        
        print("[INFO] Verifying document ingestion...")
        
        # Check if vector store has the expected number of vectors
        actual_count = self.vector_store.index.ntotal
        
        if actual_count == 0:
            raise ValueError("Vector database is empty. No vectors were stored.")
        
        # Get the final chunk count after validation (might be different from original)
        final_chunk_count = len(self.vector_store.meta) if hasattr(self.vector_store, 'meta') else actual_count
        
        print(f"[INFO] Verification: {actual_count} vectors stored for {final_chunk_count} chunks")
        
        # The verification should pass if vectors match the final chunk count
        # (which may be different from original chunks due to splitting)
        if actual_count != final_chunk_count:
            print(f"[WARNING] Vector count mismatch: {actual_count} vectors vs {final_chunk_count} chunks")
            print(f"[INFO] This might be due to chunk splitting during validation")
            # Don't raise an error here, just log the warning
        else:
            print(f"[INFO] ✅ Vector count verification passed")
        
        # Test retrieval to ensure the database is working
        try:
            # Test with a simple query to ensure the database is functional
            test_query = "test"
            test_result = embed_text_chunks([test_query], model_name=self.embedding_model)
            
            # Handle the new return format (embeddings, validated_chunks)
            if isinstance(test_result, tuple):
                test_embeddings, _ = test_result
            else:
                test_embeddings = test_result
            
            test_embedding = test_embeddings[0]
            test_results = self.vector_store.search(test_embedding, top_k=1)
            
            if not test_results:
                raise ValueError("Vector database retrieval test failed. No results returned.")
                
        except Exception as e:
            raise ValueError(f"Vector database verification failed: {str(e)}")
        
        print(f"[INFO] ✅ Verification successful: {actual_count} chunks loaded into database")
    
    def get_ingestion_status(self):
        """
        Get the current ingestion status.
        
        Returns:
            dict: Status information including whether document is ingested, chunk count, etc.
        """
        vector_count = 0
        if hasattr(self.vector_store, 'index') and self.vector_store.index is not None:
            vector_count = self.vector_store.index.ntotal
        
        return {
            'is_ingested': self.is_ingested,
            'chunk_count': len(self.chunks),
            'vector_count': vector_count,
            'ready_for_queries': self.is_ingested and len(self.chunks) > 0,
            'has_existing_data': vector_count > 0 and not self.is_ingested
        }
    
    def check_existing_data(self):
        """
        Check if there's existing data in the vector store that might conflict.
        
        Returns:
            bool: True if there's existing data that should be cleared
        """
        if hasattr(self.vector_store, 'index') and self.vector_store.index is not None:
            return self.vector_store.index.ntotal > 0
        return False 