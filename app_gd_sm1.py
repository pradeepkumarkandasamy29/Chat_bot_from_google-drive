import streamlit as st
import requests
import os

from PyPDF2 import PdfReader
from gpt4all import GPT4All


# -------------------------------
# CONFIG
# -------------------------------

FILE_ID = "18k3cQyHnFl2jh-7C8vPPzKG5pnfKJGeW"

DOWNLOAD_DIR = "pdfs"
PDF_PATH = os.path.join(DOWNLOAD_DIR, "data.pdf")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# -------------------------------
# DOWNLOAD PDF
# -------------------------------

@st.cache_data
def download_pdf():

    if os.path.exists(PDF_PATH):
        return PDF_PATH

    session = requests.Session()

    url = "https://drive.google.com/uc?export=download"
    params = {"id": FILE_ID}

    response = session.get(url, params=params, stream=True)

    # Handle Google Drive confirmation
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            params["confirm"] = value
            response = session.get(url, params=params, stream=True)
            break

    with open(PDF_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    return PDF_PATH


# -------------------------------
# EXTRACT PDF TEXT
# -------------------------------

@st.cache_data
def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        t = page.extract_text()

        if t:
            text += t + "\n"

    return text


# -------------------------------
# LOAD SMALL LLM
# -------------------------------

@st.cache_resource
def load_llm():

    llm = GPT4All(
        model_name="orca-mini-3b-gguf2-q4_0.gguf",
        model_path=".",
        allow_download=True
    )

    return llm


# -------------------------------
# ASK QUESTION
# -------------------------------

def ask_question(question, context, llm):

    # limit context for speed
    context = context[:3000]

    prompt = f"""
Answer using the PDF content below.

PDF Content:
{context}

Question:
{question}

Answer:
"""

    response = llm.generate(
        prompt,
        max_tokens=60,
        temp=0.2
    )

    return response.strip()


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="Fast PDF Chatbot",
    page_icon="🤖"
)

st.title("🏍️ Ather Energy Chatbot from Google Drive")

with st.spinner("Loading..."):

    pdf_path = download_pdf()

    pdf_text = extract_text(pdf_path)

    llm = load_llm()


question = st.text_input("Ask question :")

if question:

    with st.spinner("Thinking..."):

        answer = ask_question(
            question,
            pdf_text,
            llm
        )

    st.success(answer)