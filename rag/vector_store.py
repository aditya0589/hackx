import faiss
import numpy as np
import os
import pickle

class FaissVectorStore:
    def __init__(self, dim, index_path='faiss.index', meta_path='faiss_meta.pkl'):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []
    
    def clear(self):
        """
        Clear all vectors and metadata from the store.
        """
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []
        self.save()
        print(f"[INFO] Vector store cleared. New index created with dimension {self.dim}")
    
    def reset(self):
        """
        Reset the vector store by clearing and recreating the index.
        """
        self.clear()
        # Remove existing files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        print("[INFO] Vector store files removed and reset")

    def add(self, embeddings, metadatas):
        arr = np.array(embeddings).astype('float32')
        self.index.add(arr)
        self.meta.extend(metadatas)
        self.save()

    def search(self, query_embedding, top_k=5):
        arr = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(arr, top_k)
        results = [self.meta[i] for i in I[0]]
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.meta, f) 