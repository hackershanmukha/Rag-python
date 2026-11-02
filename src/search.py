import os
from dotenv import load_dotenv

from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b",
        groq_api_key: str = None
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        # Load or build vectorstore
        faiss_path = os.path.join(
            persist_dir, "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir, "metadata.pkl"
        )

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:
            self.vectorstore.load()

        # Initialize Groq LLM
        active_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        self.llm = ChatGroq(
            groq_api_key=active_api_key,
            model_name=llm_model
        )

        print(
            f"[INFO] Groq LLM initialized: {llm_model}"
        )

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Step 1: Retrieve relevant documents
        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        # Step 2: Extract retrieved text
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        # Step 3: Combine retrieved documents into context
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        # Step 4: Display retrieved context
        print("\n" + "=" * 60)
        print("RETRIEVED CONTEXT")
        print("=" * 60)
        print(context)

        # Step 5: Create prompt
        prompt = f"""You are a helpful assistant.

Answer the question using the following context.
If the answer is not available in the context,
say that it is not available.

Context:
{context}

Question:
{query}

Answer:"""

        # Step 6: Display final prompt
        print("\n" + "=" * 60)
        print("FINAL PROMPT (CONTEXT + QUESTION)")
        print("=" * 60)
        print(prompt)

        # Step 7: Send prompt to LLM
        response = self.llm.invoke(prompt)

        # Step 8: Display generated answer
        print("\n" + "=" * 60)
        print("LLM GENERATED ANSWER")
        print("=" * 60)
        print(response.content)

        return response.content

    def search_with_sources(
        self,
        query: str,
        top_k: int = 5
    ) -> dict:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]
        context = "\n\n".join(texts)

        if not context:
            return {
                "answer": "No relevant documents found.",
                "sources": []
            }

        prompt = f"""You are a helpful assistant.

Answer the question using the following context.
If the answer is not available in the context,
say that it is not available.

Context:
{context}

Question:
{query}

Answer:"""
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content,
            "sources": results
        }