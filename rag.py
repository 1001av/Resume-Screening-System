import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from embedder import get_embedding
from openai import OpenAI
from parser import load_resumes

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your environment or .env file."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

workspace_dir = Path(__file__).resolve().parent
vectorstore_dir = workspace_dir / "VectorStore"
vectorstore_dir.mkdir(parents=True, exist_ok=True)
vectorstore_index_path = vectorstore_dir / "index.faiss"

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - chunk_overlap

    return chunks

index = None
texts = None


def build_index():
    global index, texts
    if index is not None and texts is not None:
        return

    documents = load_resumes()
    texts = []
    for document in documents:
        texts.extend(split_text(document.page_content))
    embeddings = np.array([get_embedding(t) for t in texts]).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(vectorstore_index_path))


def retrieve(query, k=5):
    build_index()
    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)

    return [texts[i] for i in indices[0]]


def generate_answer(query, context):
    prompt = f"""
You are a recruiter assistant.

Given resumes below, find best candidates.

Context:
{context}

Query:
{query}

Return:
- Top candidates
- Skills match
- Experience match
- Reason
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def ask(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join(retrieved_docs)

    return generate_answer(query, context)