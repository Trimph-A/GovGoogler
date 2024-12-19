import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Mock documents (for indexing)
documents = [
    "How to apply for a driver's license?",
    "Steps to register a vehicle in the USA",
    "How to get a state ID card?"
]

# Index the documents
document_embeddings = model.encode(documents)
document_embeddings = np.array(document_embeddings).astype('float32')
index.add(document_embeddings)

def search_faiss(query):
    # Embed the query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Log the query embedding
    print(f"Query embedding: {query_embedding}")

    # Perform the search
    D, I = index.search(query_embedding, k=3)

    # Log the search results
    print(f"Distances: {D}")
    print(f"Indices: {I}")

     # Fetch actual document content using the indices
    results = [documents[i] for i in I[0]]  # Fetch the actual content based on indices
    
    return I  # Indices of the closest documents

# Test the search function
query = "How do I apply for a driver's license?"
results = search_faiss(query)
print(f"Top results: {results}")
