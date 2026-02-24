🧠 BrainVault
A powerful RAG (Retrieval-Augmented Generation) system that lets you upload documents and chat with them using AI. Supports both digital PDFs and handwritten/scanned documents.
✨ Features

📄 Document Upload — Upload PDF, TXT, and Markdown files via drag-and-drop
🔍 Hybrid Retrieval — Combines dense (FAISS) and sparse (BM25) search for accurate results
🤖 AI-Powered Answers — Uses DeepSeek R1 via OpenRouter for intelligent responses
📸 OCR Support — Sarvam AI Vision reads handwritten and scanned PDFs
💬 Web Chat Interface — Clean, modern UI with source citations
⚡ Background Indexing — Index rebuilds in the background without freezing the UI
🗑️ Document Management — View and delete uploaded documents
 Getting Started
Prerequisites

Python 3.10+
Tesseract OCR (optional, for basic OCR)
Poppler (for PDF to image conversion)

*Installation*

Clone the repository

git clone https://github.com/sarthak742/brainvault.git
cd brainvault

Create a virtual environment

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

Install dependencies

pip install -r requirements.txt

Set up environment variables

Create a .env file in the project root:
envOPENROUTER_API_KEY=your_openrouter_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here

Get OpenRouter API key at: https://openrouter.ai
Get Sarvam API key at: https://dashboard.sarvam.ai


Build the initial index (optional, if you have existing docs)

python build_index.py

Run the web app

python app.py
Open your browser at http://localhost:8000



Usage
Web Interface

Open http://localhost:8000
Upload documents using the sidebar (drag-and-drop or click)
Wait for the index to rebuild (status indicator in bottom left)
Ask questions in the chat interface
Get answers with source citations
