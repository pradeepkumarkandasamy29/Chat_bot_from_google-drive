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

TEXT_CACHE = "cached_text.txt"
INDEX_FILE = "faiss.index"


# -------------------------------
# GET FILE IDS
# -------------------------------

@st.cache_data
def get_pdf_links():
    response = requests.get(FOLDER_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    html = str(soup)

    file_ids = re.findall(r'([a-zA-Z0-9_-]{25,})', html)
    return list(set(file_ids))


# -------------------------------
# LOAD & CACHE PDFs
# -------------------------------

@st.cache_data
def load_all_pdfs():

    # ✅ If cached → load instantly
    if os.path.exists(TEXT_CACHE):
        with open(TEXT_CACHE, "r", encoding="utf-8") as f:
            return f.read()

    file_ids = get_pdf_links()

    all_text = ""

    for file_id in file_ids:
        try:
            pdf_url = f"https://drive.google.com/uc?id={file_id}"
            response = requests.get(pdf_url)

            pdf_data = io.BytesIO(response.content)
            reader = PdfReader(pdf_data)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

        except Exception as e:
            print(f"Skipping {file_id}: {e}")

    # ✅ Save cache
    with open(TEXT_CACHE, "w", encoding="utf-8") as f:
        f.write(all_text)

    return all_text


# -------------------------------
# CHUNKING (optimized)
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

    # ✅ lighter & faster model
    llm = GPT4All("ggml-gpt4all-j-v1.3-groovy.bin")

    return embed_model, llm


# -------------------------------
# BUILD / LOAD FAISS INDEX
# -------------------------------

@st.cache_resource
def build_index(chunks, embed_model):

    # ✅ Load if exists
    if os.path.exists(INDEX_FILE):
        return faiss.read_index(INDEX_FILE)

    vectors = embed_model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # ✅ Save index
    faiss.write_index(index, INDEX_FILE)

    return index


# -------------------------------
# CONTEXT LIMITER (optimized)
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
# QUESTION ANSWERING
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

st.title("🤖 GD PDF Chatbot")
st.write("Reads PDFs from Google Drive folder")


# -------------------------------
# LOAD DATA
# -------------------------------

with st.spinner("Loading PDFs (first run may take time)..."):
    text = load_all_pdfs()
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