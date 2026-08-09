import os
import json
import requests
import numpy as np
from ollama import Client

HF_API_KEY = os.environ.get("HF_API_KEY")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/pipeline/feature-extraction"
hf_headers = {"Authorization": f"Bearer {HF_API_KEY}"}

ollama_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

with open("chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_query_embedding(text):
    response = requests.post(EMBED_URL, headers=hf_headers, json={"inputs": text})
    return response.json()

def find_top_chunks(question, top_n=2):
    q_embedding = get_query_embedding(question)
    scored = []
    for chunk in chunks:
        score = cosine_similarity(q_embedding, chunk["embedding"])
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_n]]

def ask(question):
    top_chunks = find_top_chunks(question)
    context = "\n\n".join(c["text"] for c in top_chunks)

    prompt = f"""Use the following context to answer the question. Answer in the same language as the question.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama_client.chat(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content

if __name__ == "__main__":
    question = "What is the pH of pure water?"
    print("Q:", question)
    print("A:", ask(question))
