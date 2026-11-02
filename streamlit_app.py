import os
import shutil
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="rag-python | Document Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #2e3440;
        color: #d8dee9;
        margin-right: 0.4rem;
        margin-bottom: 0.6rem;
    }
    .source-box {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_reloaded" not in st.session_state:
    st.session_state.index_reloaded = 0


@st.cache_resource(show_spinner=False)
def get_rag_engine(model_name: str, api_key: str, cache_buster: int):
    """Instantiate and cache RAGSearch engine."""
    return RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model=model_name,
        groq_api_key=api_key,
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    # API Key Input
    default_api_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_api_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )

    # Model Selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "mixtral-8x7b-32768",
    ]
    selected_model = st.selectbox(
        "LLM Model",
        options=model_options,
        index=0,
        help="Model executed on Groq LPU engine.",
    )

    # Top-K Retrieval Slider
    top_k = st.slider(
        "Context Chunks (Top-K)",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of most similar text chunks retrieved from FAISS.",
    )

    st.markdown("---")
    st.subheader("📁 Document Management")

    # Upload new documents
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        save_btn = st.button("📥 Save Uploaded Documents", use_container_width=True)
        if save_btn:
            saved_count = 0
            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file.name).suffix.lower()
                target_dir = Path("data/pdf_files") if ext == ".pdf" else Path("data/text_files")
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / uploaded_file.name

                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1

            st.success(f"Saved {saved_count} file(s) to data/ folder!")

    # Check existing data documents
    data_path = Path("data")
    existing_docs = list(data_path.glob("**/*.*")) if data_path.exists() else []
    valid_docs = [
        f for f in existing_docs if f.suffix.lower() in [".pdf", ".txt"] and "vector_store" not in str(f)
    ]
    st.caption(f"📊 Currently found {len(valid_docs)} source file(s) in `data/`.")

    # Rebuild Index Button
    rebuild_btn = st.button("🔄 Rebuild FAISS Vector Store", use_container_width=True)
    if rebuild_btn:
        with st.spinner("Loading documents, chunking, and embedding with FAISS..."):
            try:
                docs = load_all_documents("data")
                if not docs:
                    st.error("No documents found in `data/` to index.")
                else:
                    store = FaissVectorStore("faiss_store")
                    store.build_from_documents(docs)
                    st.session_state.index_reloaded += 1
                    # Clear resource cache to pick up updated vector index
                    st.cache_resource.clear()
                    st.success("✅ FAISS index rebuilt successfully!")
            except Exception as e:
                st.error(f"Failed to build vector store: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Chat Application
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ rag-python Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Fast Retrieval-Augmented Generation with local FAISS vector search and Groq LPU inference.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.markdown('<span class="badge-pill">🔍 Vector DB: FAISS</span>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<span class="badge-pill">🧠 Embeddings: all-MiniLM-L6-v2</span>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<span class="badge-pill">⚡ LLM: {selected_model}</span>', unsafe_allow_html=True)

# Guard against missing API key
if not groq_api_key:
    st.warning("⚠️ Please provide a Groq API Key in the left sidebar or in `.env` to start chatting.")
    st.stop()

# Initialize Engine
try:
    rag_engine = get_rag_engine(
        model_name=selected_model,
        api_key=groq_api_key,
        cache_buster=st.session_state.index_reloaded,
    )
except Exception as e:
    st.error(f"Error initializing RAG engine: {e}")
    st.stop()

# Display Welcome / Starter queries if history is empty
if not st.session_state.messages:
    st.info("💡 **Welcome!** Ask any question about your indexed documents below.")
    st.markdown("##### Quick example questions:")
    example_cols = st.columns(3)
    examples = [
        "What is Association Rule Mining?",
        "Explain Apriori algorithm support and confidence.",
        "Summarize the key machine learning concepts.",
    ]
    for idx, ex in enumerate(examples):
        with example_cols[idx]:
            if st.button(ex, key=f"ex_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Retrieved Context Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    dist = src.get("distance", "N/A")
                    text = src.get("metadata", {}).get("text", "")
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <b>Source Chunk #{i}</b> | <i>Distance: {dist:.4f}</i><br>
                            {text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle New User Input
user_query = st.chat_input("Ask a question about your documents...")

# Check if query was submitted via chat input or starter example
pending_prompt = None
if user_query:
    pending_prompt = user_query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    # If a prompt was added by a button
    pending_prompt = st.session_state.messages[-1]["content"]

if pending_prompt:
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching FAISS & generating answer with Groq..."):
            try:
                res = rag_engine.search_with_sources(pending_prompt, top_k=top_k)
                answer = res["answer"]
                sources = res["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Retrieved Context Sources ({len(sources)} chunks)"):
                        for i, src in enumerate(sources, 1):
                            dist = src.get("distance", 0.0)
                            text = src.get("metadata", {}).get("text", "")
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <b>Source Chunk #{i}</b> | <i>Distance: {dist:.4f}</i><br>
                                    {text}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                st.error(f"Error generating answer: {e}")
