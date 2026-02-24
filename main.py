import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Component Imports
from embeddings.embeddings import Embedder
from vectorstore.index import VectorStore
from retrieval.retriever import Retriever
from retrieval.bm25_retrieval import BM25Retriever
from llm.client import OpenRouterClient
from llm.answer_engine import AnswerEngine
from config import get_vectorstore_dir, get_project_root

# --- Configuration ---
# Configure logging to be quiet for libraries, but visible for app errors
logging.basicConfig(
    level=logging.WARNING, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RAG_APP")
logger.setLevel(logging.INFO)

def check_api_key() -> bool:
    """Ensure OpenRouter API key is present."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n❌ Error: OPENROUTER_API_KEY not found in environment.")
        print("   Please set it before running:")
        print("   Option 1: Create a .env file with OPENROUTER_API_KEY=your_key")
        print("   Option 2: Set environment variable:")
        print("     PowerShell: $env:OPENROUTER_API_KEY='sk-or-your-key-here'")
        print("     Bash: export OPENROUTER_API_KEY='sk-or-your-key-here'\n")
        return False
    return True

def format_citation(chunk: dict) -> str:
    """Helper to format source string from chunk record."""
    source = Path(chunk.get('source', 'Unknown')).name
    start = chunk.get('start_page')
    end = chunk.get('end_page')
    
    if start is None:
        return f"{source}"
    if end is None or start == end:
        return f"{source} (Page {start})"
    return f"{source} (Pages {start}-{end})"

def main():
    # 1. Pre-flight Checks
    if not check_api_key():
        return

    # Use config for paths
    VECTORSTORE_DIR = get_vectorstore_dir()
    INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
    META_PATH = VECTORSTORE_DIR / "metadata.json"

    # 2. Verify Data Existence
    if not INDEX_PATH.exists() or not META_PATH.exists():
        logger.error(f"❌ Index not found at {VECTORSTORE_DIR}")
        print("   Run 'python build_index.py' first to ingest your documents.")
        return

    # 3. Initialize Components (The Cold Start)
    print("🧠 Loading Second Brain... (This may take a moment)")

    try:
        # Load Knowledge Base
        store = VectorStore.load(str(INDEX_PATH), str(META_PATH))

        # Extract chunks from metadata for BM25
        chunks = list(store.metadata.values())

        # Initialize embedder
        embedder = Embedder()

        # Initialize BM25 retriever for hybrid search
        bm25_retriever = BM25Retriever(chunks) if chunks else None

        # Connect Organs - Pass BM25 as sparse retriever for hybrid retrieval
        retriever = Retriever(embedder, store, sparse_retriever=bm25_retriever)
        client = OpenRouterClient() # Uses env var
        engine = AnswerEngine(retriever, client)
        
    except Exception as e:
        logger.exception(f"❌ Critical Startup Error: {e}")
        return

    print("✅ System Online. Ask me anything about your documents.")
    print("   (Type 'exit', 'quit', or 'q' to stop)\n")

    # 4. Interactive Chat Loop
    while True:
        try:
            query = input("You: ").strip()
            
            # Exit Conditions
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            if not query:
                continue

            print("Thinking...", end="\r")

            # --- GENERATION ---
            # Call the engine
            result = engine.generate_answer(query)
            
            # Clear status line
            print(" " * 20, end="\r")

            # --- DISPLAY ---
            print(f"🤖 AI: {result['answer']}\n")

            # Show citations ONLY if grounded (relevant docs found)
            if result['grounded'] and result['citations']:
                print("--- Sources ---")
                for score, chunk in result['citations']:
                    citation_str = format_citation(chunk)
                    print(f"• [{score:.2f}] {citation_str}")
                print("-" * 30 + "\n")
            elif not result['grounded']:
                print("(No relevant sources found above threshold)\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Runtime Error: {e}")
            print("❌ An error occurred. Check logs for details.\n")

if __name__ == "__main__":
    main()