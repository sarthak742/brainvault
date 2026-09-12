"""
List the models your NVIDIA NIM key can actually reach.

Run this once after putting NVIDIA_API_KEY in .env:
    python list_nvidia_models.py

Model IDs change as NVIDIA updates the catalog, so read them from the API
rather than copying from a blog post. Look for vision/VLM entries for OCR.
"""
import os, sys, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_llm_api_key

key = get_llm_api_key()
if not key:
    sys.exit("NVIDIA_API_KEY not set in .env")

r = requests.get(
    "https://integrate.api.nvidia.com/v1/models",
    headers={"Authorization": "Bearer " + key},
    timeout=30,
)
r.raise_for_status()
models = sorted(m["id"] for m in r.json().get("data", []))

print("{} models reachable.\n".format(len(models)))

hints = ("vl", "vision", "ocr", "parse", "nemoretriever")
vision = [m for m in models if any(h in m.lower() for h in hints)]

print("--- VISION / OCR CANDIDATES ---")
for m in vision or ["(none matched - check build.nvidia.com/models)"]:
    print("  " + m)

print("\n--- ALL MODELS ---")
for m in models:
    print("  " + m)
