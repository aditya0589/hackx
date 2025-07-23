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