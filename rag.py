import faiss
import numpy as np
from parser import load_resumes
from embedder import get_embedding

documents = load_resumes()

texts = [doc.page_content for doc in documents]

embeddings = np.array([get_embedding(t) for t in texts]).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index (important)
faiss.write_index(index, "vectorstore/index.faiss")

# Retrieval Logic
def retrieve(query, k=5):
    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)
    
    return [texts[i] for i in indices[0]]

# LLM Integration
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

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

# Combine Retrieval + LLm
def ask(query):
    retrieved_docs = retrieve(query)
    context = "\n\n".join(retrieved_docs)
    
    return generate_answer(query, context)