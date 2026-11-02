# rag-python ⚡

A fast, modular, and lightweight **Retrieval-Augmented Generation (RAG)** pipeline built with **Python**, **FAISS**, **Sentence Transformers**, and **Groq**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Features

- **⚡ Blazing Fast Generation**: Powered by Groq's high-speed LPU inference engine (`ChatGroq`).
- **🔍 Local Vector Search**: Uses Meta's FAISS (`IndexFlatL2`) for efficient nearest-neighbor search without requiring paid cloud vector databases.
- **🧠 Local Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) for free, offline embedding generation.
- **📄 Multi-Format Ingestion**: Supports loading and parsing both `.txt` and `.pdf` files.
- **🧩 Clean & Modular Architecture**: Distinct modules for document loading, text chunking, embedding, vector storage, and generation.
- **💾 Persistent Indexing**: Automatically caches FAISS indexes and metadata locally, avoiding redundant re-indexing on subsequent runs.

---

## 🏗️ Architecture

```
                       ┌──────────────────────┐
                       │  Documents (PDF/TXT) │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │     Data Loader      │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │  Chunk & Embeddings  │ (Sentence Transformers)
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │   FAISS Vector Store │ ◄── [faiss.index & metadata.pkl]
                       └──────────┬───────────┘
                                  │
[User Query] ────────► [Similarity Search (Top-k)]
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Augmented Prompt     │
                       │ (Context + Query)    │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │    Groq LLM Engine   │
                       └──────────┬───────────┘
                                  │
                                  ▼
                             [Final Answer]
```

---

## 📂 Project Structure

```text
rag-python/
├── data/
│   ├── pdf_files/           # Place your PDF documents here
│   └── text_files/          # Place your TXT documents here
├── faiss_store/             # Auto-generated FAISS index & metadata
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Recursively reads PDF & TXT documents
│   ├── embedding.py         # Recursive text splitting & embedding generation
│   ├── vectorstore.py       # FAISS indexing, persistence, and similarity search
│   └── search.py            # RAG orchestration: retrieval + prompt + Groq LLM
├── .env.example             # Template for required environment variables
├── .gitignore               # Keeps secrets, caches, and vector dumps out of git
├── app.py                   # CLI entry point to test queries
├── streamlit_app.py         # Full interactive Streamlit Web UI
├── pyproject.toml           # Project config and dependencies
└── requirements.txt         # Pip dependency list
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- A **Groq API Key** (Get one for free at [console.groq.com](https://console.groq.com/keys))

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/rag-python.git
cd rag-python
```

### 3. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 💡 Usage

### Option 1: Interactive Streamlit Web UI (Recommended) 🖥️

Launch the interactive chat interface in your browser:
```bash
streamlit run streamlit_app.py
```
**UI Features**:
- 💬 **Interactive Chat**: Ask questions conversationally with instant answers.
- 📤 **Document Uploader**: Upload `.pdf` and `.txt` files directly from the browser.
- 🔄 **One-Click Re-indexing**: Rebuild the FAISS index with a single click.
- 📚 **Source Inspector**: Expandable citations showing distance metrics and retrieved text chunks.
- ⚙️ **Custom Model Picker**: Switch between Groq models (Llama 3.3, Llama 3.1, Mixtral, etc.) on the fly.

### Option 2: Command Line (CLI) ⌨️

Place your documents into `data/pdf_files/` or `data/text_files/`, then run:
```bash
python app.py
```

On first run, the system will:
1. Load all files from `data/`.
2. Chunk and compute embeddings.
3. Save the FAISS index locally to `faiss_store/`.
4. Run your test query, retrieve relevant context, and output the Groq-generated response.

On subsequent runs, it detects the existing index and loads it instantly!

---

## ⚙️ Customization

You can easily configure parameters in [`src/search.py`](file:///src/search.py) or [`app.py`](file:///app.py):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Sentence transformer model for vector representations |
| `llm_model` | `"openai/gpt-oss-20b"` | LLM model deployed on Groq (e.g. `llama-3.3-70b-versatile`) |
| `chunk_size` | `1000` | Number of characters per text chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `top_k` | `3` or `5` | Number of relevant chunks retrieved from FAISS |

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).
