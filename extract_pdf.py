from pypdf import PdfReader

reader = PdfReader("test pdf files/test_chapter.pdf")

print(f"Number of pages: {len(reader.pages)}")
print("-" * 40)

full_text = ""
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    full_text += text + "\n"
    print(f"--- Page {i+1} ---")
    print(text)

# Save extracted text to a file so we can use it in the next step
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("-" * 40)
print("Saved to extracted_text.txt")
