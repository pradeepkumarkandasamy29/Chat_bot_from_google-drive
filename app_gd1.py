import streamlit as st
import requests
import re
import io

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from gpt4all import GPT4All


# -------------------------------
# GOOGLE DRIVE FOLDER
# -------------------------------

FOLDER_ID = "1BeywPB62BxpaY3vGbILr1RYChCJ3w2Bq"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


# -------------------------------
# GET ALL PDF FILE IDS FROM FOLDER
# -------------------------------

@st.cache_data
def get_pdf_links():
    response = requests.get(FOLDER_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    html = str(soup)

    # Find all Google Drive file IDs
    file_ids = re.findall(r'([a-zA-Z0-9_-]{25,})', html)
    unique_ids = list(set(file_ids))

    return unique_ids


# -------------------------------
# DOWNLOAD PDF AND EXTRACT TEXT
# -------------------------------

@st.cache_data
def load_all_pdfs():
    file_ids = get_pdf_links()

    all_text = ""

    for file_id in file_ids:
        try:
            pdf_url = f"https://drive.google.com/uc?id={file_id}"

            response = requests.get(pdf_url)
            pdf_data = io.BytesIO(response.content)

            reader = PdfReader(pdf_data)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            all_text += text + "\n"

        except Exception as e:
            print(f"Skipping file {file_id}: {e}")

    return all_text


# -------------------------------
# SPLIT INTO CHUNKS
# -------------------------------

@st.cache_data
def create_chunks(text, chunk_size=200, overlap=50):
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

    llm = GPT4All("mistral-7b-openorca.Q4_0.gguf")

    return embed_model, llm


# -------------------------------
# BUILD VECTOR INDEX
# -------------------------------

@st.cache_resource
def build_index(chunks, _embed_model):
    vectors = _embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index


# -------------------------------
# CONTEXT LIMITER
# -------------------------------

def trim_chunks(chunks, max_words=1500):
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

def ask_question(
    question,
    chunks,
    embed_model,
    index,
    llm,
    top_k=5
):
    q_vec = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    D, I = index.search(q_vec, k=top_k)

    matched_chunks = [
        chunks[i]
        for i in I[0]
    ]

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

    response = llm.generate(
        prompt,
        max_tokens=300
    )

    return response.strip(), matched_chunks


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="📄 PDF Drive Chatbot",
    page_icon="🤖"
)

st.title("🤖 GD PDF Chatbot")

st.write("Reads all PDFs from shared Google Drive folder")


# -------------------------------
# LOAD KNOWLEDGE BASE
# -------------------------------

with st.spinner("Loading PDFs from Google Drive..."):
    text = load_all_pdfs()

    chunks = create_chunks(text)

    embed_model, llm = load_models()

    index = build_index(
        chunks,
        embed_model
    )


# -------------------------------
# USER INPUT
# -------------------------------

question = st.text_input("Ask your question:")


# -------------------------------
# ANSWER GENERATION
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
