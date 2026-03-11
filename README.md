# RAG-Chatbot-offline
A fully offline RAG based chatbot

1. Problem I Solved

I built a Retrieval-Augmented Generation (RAG) chatbot that can answer queries based on enterprise documents like policies, procedures, and knowledge base exports.

The goal: enable employees to query PDFs, Word docs, tables, and even scanned images and get contextual answers.

It runs entirely offline for data security (important in enterprise environments).

2. Tech Stack

Python 3.11 + virtualenv for environment isolation.

Sentence Transformers (all-MiniLM-L6-v2) → for generating embeddings (vector representation of text).

FAISS → fast vector similarity search, to retrieve top-k relevant chunks.

Microsoft Phi-2 (local LLM) → lightweight model used as the response generator.

PyMuPDF + pdfplumber + Tesseract OCR → extract text, tables, and images from PDFs.

Streamlit UI (frontend) → simple web interface with chat history, config controls, and document source display.

dotenv (.env file) → configurable settings for chunk size, overlap, OCR language, etc.

3. Pipeline (How It Works)

Document Ingestion

Load files from /data/ folder (PDF, DOCX, CSV, JSON, TXT, HTML, Markdown).

Handle:

Text: extract with PyMuPDF.

Tables: extracted as CSV-like text with pdfplumber.

Images: OCR with Tesseract.

Each document is split into overlapping chunks (e.g., 800 tokens with 120 overlap) for better retrieval granularity.

Embedding & Indexing

Each chunk → converted into embeddings using all-MiniLM-L6-v2.

Stored in a FAISS index for fast semantic search.

Metadata (doc name, page number, type: text/table/image) stored in meta.jsonl.

Query Flow

User enters query in the UI.

System embeds the query → retrieves Top-K relevant chunks from FAISS.

These chunks are passed into the LLM (phi-2) as context.

LLM generates a natural language answer, grounded in retrieved chunks.

UI/Features

Clean chat interface (Streamlit).

Shows answers + source document + page number.

Configurable .env file (change chunk size, OCR language, etc.).

Supports index rebuild (--rebuild) for new documents.

