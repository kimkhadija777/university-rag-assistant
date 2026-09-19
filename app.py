import os
import json
import shutil
import zipfile
from pathlib import Path
import streamlit as st
import faiss
import gdown
from sentence_transformers import SentenceTransformer
from groq import Groq

# Page Config
st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="centered")

INDEX_DIR = Path("vector_store")
INDEX_PATH = INDEX_DIR / "university.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5
MIN_SIMILARITY = 0.25

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# Automatically prepare index files if not present on server boot
@st.cache_resource
def ensure_vector_store():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        with st.spinner("📦 First-time initialization: Building Knowledge Base vector store..."):
            try:
                # Automatic Fallback Ingestion via ingest module
                import ingest
                ingest.main()
            except Exception as e:
                st.error(f"❌ Failed to run vector ingestion automatically: {e}")
                st.stop()

@st.cache_resource
def load_faiss_and_metadata():
    ensure_vector_store()
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

st.title("🎓 University Enterprise RAG Assistant")
st.markdown("Ask anything regarding policies, course outlines, deadlines, or registration processes.")

index, all_chunks = load_faiss_and_metadata()

if not index or not all_chunks:
    st.error("⚠️ Could not locate or build the vector index. Verify Google Drive access permissions.")
    st.stop()

if not GROQ_API_KEY:
    st.warning("🔑 Groq API Key missing! Set `GROQ_API_KEY` in Streamlit Secrets.")
    st.stop()

embedding_model = load_embedding_model()
client = Groq(api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def search(query: str):
    q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) >= MIN_SIMILARITY:
            results.append({"score": float(score), **all_chunks[idx]})
    return results

if prompt := st.chat_input("Ask your university question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching university database..."):
            retrieved = search(prompt)

            if not retrieved:
                answer = "I could not find relevant information in the official university knowledge base."
                sources = []
            else:
                context_str = "\n\n".join([
                    f"SOURCE {i+1} (File: {item['metadata']['source']}):\n{item['text']}"
                    for i, item in enumerate(retrieved)
                ])

                system_prompt = f"""
You are a University Knowledge Assistant.
You answer questions using ONLY the retrieved context below.

Rules:
1. Do not invent university policies.
2. If answer is missing, clearly state that.
3. Cite source files when appropriate.

Context:
{context_str}
"""
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                answer = response.choices[0].message.content
                sources = list({item["metadata"]["source"] for item in retrieved})

            if sources:
                answer += "\n\n**📚 Sources Used:**\n" + "\n".join([f"- `{src}`" for src in sources])

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
