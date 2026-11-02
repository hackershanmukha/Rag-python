from pathlib import Path

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":

    # Initialize FAISS vector store
    store = FaissVectorStore("faiss_store")

    # Check whether index files exist
    faiss_path = Path("faiss_store/faiss.index")
    meta_path = Path("faiss_store/metadata.pkl")

    if faiss_path.exists() and meta_path.exists():

        print("[INFO] Existing FAISS index found.")
        store.load()

    else:

        print("[INFO] FAISS index not found. Building new index...")

        docs = load_all_documents("data")

        if not docs:
            raise ValueError("No documents found in the data folder.")

        store.build_from_documents(docs)

    # Test vector store search
    results = store.query(
        "What is Association Rule Mining?",
        top_k=3
    )

    print("\nSearch Results:")
    print(results)

    # Initialize RAG search
    rag_search = RAGSearch()

    query = "What is Association Rule Mining?"

    summary = rag_search.search_and_summarize(
        query,
        top_k=3
    )

    print("\nSummary:")
    print(summary)