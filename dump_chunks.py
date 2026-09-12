"""
Dump the indexed chunks to a readable file so you can write eval questions
against what the system ACTUALLY has, not what you think the PDFs contain.

45 pages went through OCR, so the indexed text differs from the original.
Writing questions from the PDFs risks asking about text the index never saw.

    python dump_chunks.py
    -> eval/corpus_readable.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_vectorstore_dir

vs = get_vectorstore_dir()
meta = json.loads((vs / "metadata.json").read_text(encoding="utf-8"))

chunks = list(meta.values()) if isinstance(meta, dict) else list(meta)

def sort_key(c):
    return (str(c.get("source", "")), c.get("start_page") or 0)

out = Path("eval/corpus_readable.txt")
out.parent.mkdir(exist_ok=True)

lines = []
current_doc = None
for c in sorted(chunks, key=sort_key):
    doc = Path(str(c.get("source", "?"))).name
    if doc != current_doc:
        current_doc = doc
        lines.append("\n" + "=" * 78)
        lines.append("DOCUMENT: " + doc)
        lines.append("=" * 78)
    pg = c.get("start_page")
    end = c.get("end_page")
    span = "p{}".format(pg) if (end is None or end == pg) else "p{}-{}".format(pg, end)
    text = " ".join(str(c.get("text", "")).split())
    lines.append("\n--- {} [{}] ({} chars) ---".format(span, c.get("chunk_id", "?"), len(text)))
    lines.append(text)

out.write_text("\n".join(lines), encoding="utf-8")
print("Wrote {} ({} chunks)".format(out, len(chunks)))
print("\nPer-document chunk counts:")
counts = {}
for c in chunks:
    counts[Path(str(c.get("source", "?"))).name] = counts.get(Path(str(c.get("source", "?"))).name, 0) + 1
for d, n in sorted(counts.items()):
    print("  {:<28} {:>4} chunks".format(d, n))
