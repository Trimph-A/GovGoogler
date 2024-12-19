from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

def embed_and_store_data(texts):
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
