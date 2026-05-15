import streamlit as st
import requests
import io
import os

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from gpt4all import GPT4All


# -------------------------------
# CONFIG
# -------------------------------

FILE_ID = "1N0qgZ48fBB6qxqrFFfTayUoXlFlG4960"

DOWNLOAD_DIR = "pdfs"
PDF_PATH = os.path.join(DOWNLOAD_DIR, "data.pdf")
TEXT_FILE = "data.txt"
INDEX_FILE = "faiss.index"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# -------------------------------
# DOWNLOAD PDF (CORRECT WAY)
# -------------------------------

def download_pdf():

    if os.path.exists(PDF_PATH):
        return PDF_PATH

    session = requests.Session()

    url = "https://drive.google.com/uc?export=download"
    params = {"id": FILE_ID}

    response = session.get(url, params=params, stream=True)

    # 🔥 Handle confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params["confirm"] = value
            response = session.get(url, params=params, stream=True)
            break

    # Save file
    with open(PDF_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    # ✅ Final validation (optional but useful)
    if os.path.getsize(PDF_PATH) < 5000:
        raise Exception("❌ Download failed or not a real PDF (too small file)")

    return PDF_PATH


# -------------------------------
# PDF → TEXT (CACHED)
# -------------------------------

@st.cache_data
def load_text(pdf_path):

    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()

    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# -------------------------------
# CHUNKING
# -------------------------------

@st.cache_data
def create_chunks(text, chunk_size=300, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------
# LOAD MODELS
# -------------------------------

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # ⚠️ Make sure model file exists locally
    llm = GPT4All(
        model_name="mistral-7b-openorca.Q4_0.gguf",
        model_path=".",
        allow_download=False
    )

    return embed_model, llm


# -------------------------------
# BUILD / LOAD INDEX
# -------------------------------

@st.cache_resource
def build_index(chunks, _embed_model):

    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)

    vectors = _embed_model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    return index


# -------------------------------
# CONTEXT LIMIT
# -------------------------------

def trim_chunks(chunks, max_words=300):
    total_words = 0
    selected = []

    for chunk in chunks:
        words = chunk.split()

        if total_words + len(words) > max_words:
            break

        selected.append(chunk)
        total_words += len(words)

    return "\n\n".join(selected)


# -------------------------------
# QA FUNCTION
# -------------------------------

def ask_question(question, chunks, embed_model, index, llm, top_k=1):

    q_vec = embed_model.encode([question], convert_to_numpy=True)

    D, I = index.search(q_vec, k=top_k)

    matched_chunks = [chunks[i] for i in I[0]]

    context = trim_chunks(matched_chunks)

    prompt = f"""
You are a smart industrial AI assistant.

Use the context below to answer accurately.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.generate(prompt, max_tokens=80)

    return response.strip(), matched_chunks


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="📄 PDF Chatbot",
    page_icon="🤖"
)

st.title("🤖  Ather Energy Google Drive PDF Chatbot")
st.write("Auto-downloads PDF → converts → answers questions")


# -------------------------------
# PIPELINE
# -------------------------------

with st.spinner("Preparing system (first run only)..."):

    pdf_path = download_pdf()
    text = load_text(pdf_path)

    chunks = create_chunks(text)

    embed_model, llm = load_models()
    index = build_index(chunks, embed_model)


# -------------------------------
# USER INPUT
# -------------------------------

question = st.text_input("Ask your question:")


# -------------------------------
# ANSWER
# -------------------------------

if question:
    with st.spinner("Thinking..."):
        answer, matched = ask_question(
            question,
            chunks,
            embed_model,
            index,
            llm
        )

    st.success(answer)

    with st.expander("Context Used"):
        for i, chunk in enumerate(matched, 1):
            st.markdown(f"### Chunk {i}")
            st.write(chunk)