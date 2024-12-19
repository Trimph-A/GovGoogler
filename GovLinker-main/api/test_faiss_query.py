from faiss_index import search_faiss

def test_faiss_query():
    query = "How do I apply for a driver's license?"
    results = search_faiss(query)
    print(f"Top results: {results}")

if __name__ == "__main__":
    test_faiss_query()
