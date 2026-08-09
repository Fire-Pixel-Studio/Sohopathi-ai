import os
from ollama import Client

api_key = os.environ.get("OLLAMA_API_KEY")

client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {api_key}"}
)

response = client.chat(
    model="gpt-oss:120b-cloud",
    messages=[{"role": "user", "content": "তুমি কি বাংলায় উত্তর দিতে পারো?"}]
)

print(response.message.content)
