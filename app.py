import json
from pathlib import Path
import streamlit as st
import faiss
import numpy as np
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

# Load API Key securely from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# Single-load resource caches to prevent reloading models on re-renders
@st.cache_resource
def load_faiss_and_metadata():
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
    st.error("⚠️ Vector index missing! Please run `python ingest.py` locally or upload pre-built index files.")
    st.stop()

if not GROQ_API_KEY:
    st.warning("🔑 Groq API Key missing! Set `GROQ_API_KEY` in Streamlit Secrets.")
    st.stop()

embedding_model = load_embedding_model()
client = Groq(api_key=GROQ_API_KEY)

# Chat Session setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Semantic Retrieval Engine
def search(query: str):
    q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) >= MIN_SIMILARITY:
            results.append({"score": float(score), **all_chunks[idx]})
    return results

# Handle User Interaction
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

            # Append Sources
            if sources:
                answer += "\n\n**📚 Sources Used:**\n" + "\n".join([f"- `{src}`" for src in sources])

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
