# 🧠 Document Intelligence Pipeline (WIP)

> **Status:** 🚧 *Development in Progress*  
> A modular pipeline for ingesting PDFs and images, performing OCR, generating embeddings, and enabling semantic search + Q&A using Qdrant and transformer models.

---

## 📘 Overview

This project provides an **end-to-end document understanding system** consisting of two major components:

1. **`ingest_service.py`** – Reads and processes documents (PDFs, images), performs OCR, extracts text, embeds the chunks, and stores them in a **Qdrant vector database**.  
2. **`query_service.py`** – Exposes a **FastAPI REST API** endpoint that lets you query the indexed data and get AI-generated answers using retrieved document chunks as context.

---

## ⚙️ Architecture

            ┌────────────────────────┐
            │   Documents (PDF/IMG)  │
            └────────────┬───────────┘
                         │
                         ▼
         ┌──────────────────────────────────┐
         │      ingest_service.py            │
         │  - OCR (DeepSeek / Tesseract / EasyOCR)
         │  - Text chunking & embeddings
         │  - Store in Qdrant                │
         └────────────────┬─────────────────┘
                          │
                          ▼
              ┌────────────────────┐
              │     Qdrant DB      │
              │ (Vector Store)     │
              └─────────┬──────────┘
                        │
                        ▼
           ┌────────────────────────────┐
           │     query_service.py        │
           │ - Embed user query          │
           │ - Retrieve similar chunks   │
           │ - Generate answer w/ context│
           └────────────────────────────┘

---

## 🧩 Features

✅ Automatic text extraction from PDFs and images  
✅ Multiple OCR backends: **DeepSeek**, **pytesseract**, **EasyOCR**  
✅ Sentence embedding generation via **SentenceTransformers**  
✅ Vector storage & semantic search using **Qdrant**  
✅ FastAPI query service for natural-language question answering  
✅ Multithreaded ingestion for faster processing  

---


## 🧾 Example `config.yaml`

```yaml
docs_dir:
  docs_to_ingest: "./docs_to_ingest"

models:
  embedding: "sentence-transformers/all-MiniLM-L6-v2"
  ocr: "deepseek-ai/deepseek-doc-ocr"
  generation: "gpt2"   # You can swap for a better model (e.g. Mistral, Phi, etc.)

qdrant:
  url: "http://localhost:6333"
  api_key: ""
  collection: "documents_index"

ingest:
  chunk_chars: 1000
  chunk_overlap: 100
  top_k: 3
```

## Installation

```bash
git clone https://github.com/shivampandey2389/IngestorX.git
cd IngestorX
```
now install the required libraries
```bash
pip install -r requirements.txt
```
or

```bash
pip install PyPDF2 pillow pyyaml sentence-transformers qdrant-client pdf2image pytesseract easyocr transformers fastapi uvicorn
```

## setup the Qdrant

Start a local Qdrant instance using Docker:

```bash
docker run -p 6333:6333 -d qdrant/qdrant
```
## Run the Ingestion Service

To ingest all documents from the docs_to_ingest/ folder:
```bash
python ingest_service.py
```

## Run the Query Service

Launch the query API with:
```bash
uvicorn query_service:app --reload --port 8000
```
Example Request

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "Summarize the financial report", "top_k": 3}'
```
Example response:
```
{
  "answer": "The financial report discusses quarterly revenue growth...",
  "retrieved": [
    {
      "source_file": "report.pdf",
      "chunk_index": 0,
      "text": "Quarterly revenue increased by..."
    },
    {
      "source_file": "report.pdf",
      "chunk_index": 1,
      "text": "The company expects growth in..."
    }
  ]
}
```


---

## 🧠 How It Works (Summary)

1. **OCR + Text Extraction**  
   Extracts readable text from PDFs and image files using multiple OCR backends — **DeepSeek**, **pytesseract**, and **EasyOCR**.

2. **Chunking**  
   Long documents are split into overlapping text chunks to make them easier to embed and retrieve semantically.

3. **Embeddings**  
   Each chunk is transformed into a dense numerical vector using a **SentenceTransformer** model.

4. **Vector Storage**  
   The embeddings and their metadata (source file, chunk index, etc.) are stored in a **Qdrant** vector database for fast semantic search.

5. **Retrieval + Generation**  
   When a user submits a query, its embedding is compared against stored vectors to find the most relevant chunks.  
   The retrieved context is then passed to a **text-generation model** to produce a natural, contextual answer.

---
