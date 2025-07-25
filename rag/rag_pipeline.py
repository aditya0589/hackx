from .document_loader import load_document
from .text_splitter import split_text
from .embedder import embed_text_chunks
from .vector_store import FaissVectorStore
from .query_optimizer import optimize_query
from .retriever import retrieve_relevant_chunks
import google.generativeai as genai

# Main RAG pipeline using Gemini 1.5 Flash for generation and embedding-001 for embeddings
class RAGPipeline:
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.vector_store = FaissVectorStore(dim=embedding_dim)
        self.chunks = []
        self.embedding_model = 'models/embedding-001'
        self.generation_model = 'models/gemini-1.5-flash'

    def ingest_document(self, link):
        text, _ = load_document(link)
        self.chunks = split_text(text)
        embeddings = embed_text_chunks(self.chunks, model_name=self.embedding_model)
        self.vector_store.add(embeddings, self.chunks)

    def answer_query(self, query):
        """
        Returns (answer, references) where references is a list of (index, chunk) tuples.
        """
        # Use generation model for query optimization
        optimized_query = optimize_query(query, model_name=self.generation_model)
        # Use embedding model for retrieval
        references = retrieve_relevant_chunks(optimized_query, self.vector_store, self.chunks, model_name=self.embedding_model)
        context = '\n'.join(chunk for _, chunk in references)
        # Generate answer using Gemini 1.5 Flash
        prompt = f"""
You are an expert assistant. Based on the context below, answer the question with:
- A direct and concise answer
- Cite specific lines from the context
- Avoid information not present in the context

Context:
{context}

Question: {optimized_query}
Answer:
"""
        model = genai.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip(), references 
