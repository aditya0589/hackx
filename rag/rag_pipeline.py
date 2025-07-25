from .document_loader import load_document
from .text_splitter import split_text
from .embedder import embed_text_chunks
from .vector_store import FaissVectorStore
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
        if not text or not text.strip():
            raise ValueError("The document contains no extractable text. Please provide a document with selectable text.")
        self.chunks = split_text(text)
        if not self.chunks:
            raise ValueError("The document could not be split into chunks. It may be empty or not processable.")
        embeddings = embed_text_chunks(self.chunks, model_name=self.embedding_model)
        if not embeddings:
            raise ValueError("No embeddings could be generated from the document. Please check the document content.")
        self.vector_store.add(embeddings, self.chunks)

    def answer_query(self, query):
        """
        Returns (answer, references) where references is a list of (index, chunk) tuples.
        """
        if not self.chunks:
            raise ValueError("No document has been ingested or the document is empty. Please ingest a valid document first.")
        # Directly use the user query for retrieval and generation
        references = retrieve_relevant_chunks(query, self.vector_store, self.chunks, model_name=self.embedding_model)
        context = '\n'.join(chunk for _, chunk in references)
        # Generate answer using Gemini 1.5 Flash
        prompt = f"""
You are an expert assistant. Based on the context, answer the question in a consize way
If the provided context is insufficient, just assume the nearest correct answer.
Context:
{context}

Question: {query}
Answer:
"""
        model = genai.GenerativeModel(self.generation_model)
        response = model.generate_content(prompt)
        return response.text.strip(), references 
