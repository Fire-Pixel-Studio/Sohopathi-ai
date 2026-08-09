import json

with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split by sections using the numbered headings (e.g. "3.1", "3.2")
import re
sections = re.split(r'\n(?=\d\.\d )', text)

chunks = []
for i, section in enumerate(sections):
    section = section.strip()
    if len(section) > 20:  # skip empty/tiny fragments
        chunks.append({
            "id": i,
            "text": section
        })

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Created {len(chunks)} chunks")
for c in chunks:
    print(f"--- Chunk {c['id']} ---")
    print(c['text'][:100] + "...")
    print()
