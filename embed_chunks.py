import os
import json
import requests

HF_API_KEY = os.environ.get("HF_API_KEY")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/pipeline/feature-extraction"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

for chunk in chunks:
    response = requests.post(API_URL, headers=headers, json={"inputs": chunk["text"]})
    result = response.json()
    if response.status_code != 200:
        print(f"Chunk {chunk['id']} error: {result}")
    chunk["embedding"] = result
    print(f"Embedded chunk {chunk['id']}")

with open("chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Saved chunks_with_embeddings.json")
