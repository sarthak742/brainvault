"""
Calibrate the grounding threshold from data.

Prints the raw cosine similarity of the best-matching chunk for questions we
KNOW are answerable from the corpus, and for questions we KNOW are not.
The threshold belongs in the gap between the two distributions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_vectorstore_dir
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore

IN_CORPUS = [
    "What is the lifespan of a CFL bulb?",
    "How many hours does an LED last?",
    "What principle does a Megger work on?",
    "Which law governs how a BLDC motor produces torque?",
    "What keeps the generator speed constant during an insulation test?",
    "What are the three main components of a BLDC motor?",
    "What is total internal reflection in an optical fibre?",
    "What is population inversion in a laser?",
    "What is the truth table of a NAND gate?",
    "What are the principles of green chemistry?",
    "What is atom economy?",
    "What is numerical aperture?",
]

OUT_OF_CORPUS = [
    "How do you calculate power factor correction for an inductive load?",
    "What is the capital of Brazil?",
    "Who won the football world cup in 1998?",
    "How do I make sourdough bread?",
    "What is the best way to train for a marathon?",
    "Explain the plot of Hamlet.",
    "What is the current price of Bitcoin?",
    "How do I file my income tax return?",
]

vs = get_vectorstore_dir()
store = VectorStore.load(str(vs / "index.faiss"), str(vs / "metadata.json"))
emb = Embedder()

def top_cosine(q):
    res = store.search(emb.embed_query(q), k=1)
    return res[0][0] if res else 0.0

print("=" * 62)
print("IN CORPUS (answer exists)")
print("=" * 62)
ins = []
for q in IN_CORPUS:
    s = top_cosine(q)
    ins.append(s)
    print("  {:.3f}  {}".format(s, q[:52]))

print()
print("=" * 62)
print("OUT OF CORPUS (no answer exists)")
print("=" * 62)
outs = []
for q in OUT_OF_CORPUS:
    s = top_cosine(q)
    outs.append(s)
    print("  {:.3f}  {}".format(s, q[:52]))

lo_in, hi_out = min(ins), max(outs)
print()
print("=" * 62)
print("in-corpus  min {:.3f}  mean {:.3f}".format(lo_in, sum(ins)/len(ins)))
print("out-corpus max {:.3f}  mean {:.3f}".format(hi_out, sum(outs)/len(outs)))
if lo_in > hi_out:
    print("SEPARABLE. Suggested threshold: {:.2f}".format((lo_in + hi_out) / 2))
else:
    print("OVERLAP of {:.3f} - no threshold separates these cleanly.".format(hi_out - lo_in))
    print("Pick by cost: higher = fewer false answers, more refusals.")
