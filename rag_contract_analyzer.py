import streamlit as st
import pdfplumber
import openai
import faiss
import numpy as np
from typing import List
from utils.embeddings_utils import get_embedding


# Config
openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"

# PDF helper
def load_pdf_text(file) -> str:
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Embedding
def embed_chunks(chunks: List[str]):
    return [get_embedding(chunk, engine=EMBEDDING_MODEL) for chunk in chunks]

# FAISS vector store
def create_faiss_index(embeddings: List[List[float]]):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

def retrieve_relevant_chunks(index, query_text, all_chunks, top_k=5):
    query_embedding = get_embedding(query_text, engine=EMBEDDING_MODEL)
    D, I = index.search(np.array([query_embedding]).astype("float32"), top_k)
    return [all_chunks[i] for i in I[0]]

# GPT-4 extraction
def extract_parameters(context_chunks: List[str], new_contract_text: str):
    prompt = f"""
You are an AI trained to extract key details from contracts.

Below are relevant context chunks from prior contracts:
{'\n---\n'.join(context_chunks)}

Now analyze the following contract and return the parameters as JSON:
{new_contract_text}

Return:
- Contract Start Date
- Contract End Date
- Parties Involved
- Payment Terms
- Jurisdiction
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("📄 Contract Analyzer with RAG")
st.markdown("Upload past contracts (to build context), then a new one to extract key terms.")

with st.sidebar:
    st.subheader("🔁 Upload Past Contracts")
    past_contracts = st.file_uploader("Upload historical contracts (PDF)", type=["pdf"], accept_multiple_files=True)

    st.subheader("📄 Upload New Contract")
    new_contract = st.file_uploader("Upload new contract to analyze", type=["pdf"])

if past_contracts and new_contract:
    with st.spinner("Processing..."):
        all_chunks = []
        for file in past_contracts:
            text = load_pdf_text(file)
            all_chunks.extend(chunk_text(text))

        embeddings = embed_chunks(all_chunks)
        index = create_faiss_index(embeddings)

        new_text = load_pdf_text(new_contract)
        new_text_chunks = chunk_text(new_text)
        new_full_text = " ".join(new_text_chunks)

        relevant_chunks = retrieve_relevant_chunks(index, new_full_text, all_chunks)
        extracted = extract_parameters(relevant_chunks, new_full_text)

    st.subheader("🧠 Extracted Parameters")
    st.code(extracted, language="json")
else:
    st.info("⬅️ Upload contracts in the sidebar to get started.")

