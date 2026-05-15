import streamlit as st
import requests
import re
import io
import os

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from gpt4all import GPT4All


# -------------------------------
# CONFIG
# -------------------------------

FOLDER_ID = "1BeywPB62BxpaY3vGbILr1RYChCJ3w2Bq"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"

DOWNLOAD_DIR = "pdfs"
TEXT_FILE = "combined.txt"
INDEX_FILE = "faiss.index"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# -------------------------------
# GET PDF IDS (SCRAPING)
# -------------------------------

@st.cache_data
def get_pdf_ids():
    response = requests.get(FOLDER_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    html = str(soup)

    file_ids = re.findall(r'([a-zA-Z0-9_-]{25,})', html)
    return list(set(file_ids))


# -------------------------------
# DOWNLOAD PDFs (ONLY NEW)
# -------------------------------

def download_pdfs():
    file_ids = get_pdf_ids()
    local_files = []

    for file_id in file_ids:
        file_path = os.path.join(DOWNLOAD_DIR, f"{file_id}.pdf")

        # Skip if already exists
        if os.path.exists(file_path):
            local_files.append(file_path)
            continue

        try:
            pdf_url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.get(pdf_url)

            with open(file_path, "wb") as f:
                f.write(response.content)

            local_files.append(file_path)

        except Exception as e:
            print(f"Download failed: {file_id} → {e}")

    return local_files


# -------------------------------
# CONVERT PDF TO TEXT (CACHED)
# -------------------------------

@st.cache_data
def convert_pdfs_to_text(pdf_files):

    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()

    all_text = ""

    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)

            for page in reader.pages:
                t = page.extract_text()
                if t:
                    all_text += t + "\n"

        except Exception as e:
            print(f"Error reading {pdf}: {e}")

    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(all_text)

    return all_text


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

    # Faster model
    llm = GPT4All("mistral-7b-openorca.Q4_0.gguf")

    return embed_model, llm


# -------------------------------
# BUILD / LOAD INDEX
# -------------------------------

@st.cache_resource
def build_index(chunks, embed_model):

    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)

    vectors = embed_model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)

    return index


# -------------------------------
# CONTEXT LIMIT
# -------------------------------

def trim_chunks(chunks, max_words=600):
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

def ask_question(question, chunks, embed_model, index, llm, top_k=2):

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

    response = llm.generate(prompt, max_tokens=200)

    return response.strip(), matched_chunks


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="📄 PDF Drive Chatbot",
    page_icon="🤖"
)

st.title("🤖 Google Drive PDF Chatbot")
st.write("Auto-downloads PDFs and builds smart search system")


# -------------------------------
# LOAD PIPELINE
# -------------------------------

with st.spinner("Preparing system (first run may take time)..."):

    pdf_files = download_pdfs()                  # ✅ Auto download
    text = convert_pdfs_to_text(pdf_files)       # ✅ Cached text

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