import os
import re
import json
import shutil
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import faiss
import gdown
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# Setup Paths
BASE_DIR = Path("data_store")
DOWNLOAD_DIR = BASE_DIR / "download"
SOURCE_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = Path("vector_store")

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Updated Naya Google Drive Link
DRIVE_URL = "https://drive.google.com/file/d/16k8Vpi7g8q7p9KH0YoCOCmlOBMSk4QNG/view?usp=drivesdk"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def extract_drive_file_id(url: str) -> str:
    patterns = [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"/d/([a-zA-Z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid Google Drive URL.")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_document(path: Path) -> str:
    ext = path.suffix.lower()
    text = ""
    if ext == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            p = page.extract_text() or ""
            if p.strip():
                pages.append(f"[Page {i}]\n{p}")
        text = "\n\n".join(pages)
    elif ext == ".docx":
        doc = Document(str(path))
        text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    return clean_text(text)

def split_into_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def create_chunks(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        for s in split_into_sentences(p):
            if not current:
                current = s
            elif len(current) + len(s) + 1 <= CHUNK_SIZE:
                current += " " + s
            else:
                chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:] + " " + s
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c.strip()) >= 40]

def main():
    print("📥 Step 1: Downloading from Google Drive...")
    file_id = extract_drive_file_id(DRIVE_URL)
    download_path = DOWNLOAD_DIR / "knowledge_base"
    
    if download_path.exists():
        if download_path.is_file(): download_path.unlink()
        else: shutil.rmtree(download_path)
        
    downloaded = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(download_path), quiet=False)
    downloaded_file = Path(downloaded)

    # File Extraction
    if zipfile.is_zipfile(downloaded_file):
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(SOURCE_DIR)
    elif downloaded_file.suffix.lower() in [".pdf", ".txt", ".docx"]:
        shutil.copy2(downloaded_file, SOURCE_DIR / downloaded_file.name)

    # Document Finding
    docs = [f for f in SOURCE_DIR.rglob("*") if f.is_file() and f.suffix.lower() in {".txt", ".pdf", ".docx"}]
    print(f"📄 Found {len(docs)} document(s). Processing...")

    all_chunks = []
    for doc in docs:
        raw_text = extract_document(doc)
        if not raw_text: continue
        chunks = create_chunks(raw_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc.name,
                    "chunk_id": len(all_chunks),
                    "chunk_number": i + 1,
                    "total_chunks_in_document": len(chunks)
                }
            })

    print(f"✂️ Total chunks extracted: {len(all_chunks)}")

    print("🧠 Generating Embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [item["text"] for item in all_chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    print("🗄️ Indexing into FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save Stores
    faiss.write_index(index, str(INDEX_DIR / "university.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("✅ Ingestion Completed Successfully!")

if __name__ == "__main__":
    main()
