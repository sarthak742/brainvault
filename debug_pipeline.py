from ingestion.ingest import load_documents, extract_text_from_pdf, extract_text_from_txt, extract_text_from_markdown
from chunking.chunker import chunk_documents
from pathlib import Path

# ---- Run Ingestion ----
DATA_ROOT = Path("data")

files = load_documents(DATA_ROOT)

page_records = []

for f in files:
    if f.suffix.lower() == ".pdf":
        page_records.extend(extract_text_from_pdf(f))
    elif f.suffix.lower() == ".txt":
        page_records.extend(extract_text_from_txt(f))
    elif f.suffix.lower() == ".md":
        page_records.extend(extract_text_from_markdown(f))

print(f"\nTOTAL PAGE RECORDS: {len(page_records)}")

# ---- Run Chunking ----
chunks = chunk_documents(page_records)

print(f"TOTAL CHUNKS: {len(chunks)}")

# ---- Inspect Samples ----
print("\n=== SAMPLE CHUNKS ===")
for i, c in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i+1} ---")
    print("Source:", c["source"])
    print("Pages:", c["start_page"], "→", c["end_page"])
    print("Length:", len(c["text"]))
    print("Text Preview:")
    print(c["text"][:500])
