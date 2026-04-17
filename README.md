# 🧠 Research Assistant - RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions from a set of research papers.

Instead of relying only on a language model, the system retrieves relevant context from documents using vector search and generates answers grounded in that context.

The focus of this project is to understand how real-world LLM systems work beyond simple API calls - combining document processing, embeddings, retrieval, and generation into a complete pipeline.

---

## Papers Used

The knowledge base consists of the following research papers:

* Attention Is All You Need
* BERT: Pre-training of Deep Bidirectional Transformers
* Retrieval-Augmented Generation (RAG)
* LLaMA 2

These papers are stored locally and used to answer user queries.

---

## Tech Stack

* Python
* LangChain
* ChromaDB (vector database)
* Sentence Transformers (embeddings)
* Groq (LLM - llama-3.1-8b-instant)
* Streamlit (UI)
* PyMuPDF (PDF parsing)

---

## How It Works

### 1. Document Loading

* PDFs are loaded from the `papers/` folder using PyMuPDF

### 2. Chunking

* Documents are split into smaller chunks
* Chunk size: 500
* Overlap: 100

### 3. Embeddings

* Each chunk is converted into a vector using:

  * `all-MiniLM-L6-v2`

### 4. Vector Storage

* Embeddings are stored in ChromaDB
* Persisted locally in `chroma_db/`

### 5. Retrieval

* Top 5 relevant chunks are retrieved using similarity search

### 6. Generation

* Retrieved context is passed to Groq LLM
* Model used:

  * `llama-3.1-8b-instant`
* Custom prompt ensures grounded answers

---

## Prompt Design

The model is instructed to:

* Answer only using provided context
* Avoid hallucination
* Say "I don't have enough information..." if context is insufficient

---

## Features

* Chat-based interface (Streamlit)
* Local vector database (ChromaDB)
* Fast inference using Groq
* Source tracking for retrieved documents
* Persistent embeddings (no rebuild required every run)

---

## Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd <project-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

Make sure the `papers/` folder contains the PDFs.

Run the app:

```bash
streamlit run app.py
```

---

## Project Structure

```
project/
 ├── app.py
 ├── rag_pipeline.py
 ├── papers/
 ├── chroma_db/ (auto-generated)
 ├── requirements.txt
 ├── .env (not included)
```

---

## Example

<img width="400" height="877" alt="Screenshot 2026-04-17 110133" src="https://github.com/user-attachments/assets/ce646e73-1eb6-4dfc-9cd5-4db2030786b9" />


---

## Limitations

* Basic similarity search (no reranking)
* No conversational memory
* No real-time streaming from LLM
* Performance depends on chunking and embedding quality

---

## Future Improvements

* Add reranking (improve retrieval quality)
* Add conversational memory
* Improve prompt for better reasoning
* Replace fake streaming with real token streaming

---

## Why This Project

This project was built to move from basic ML models to real-world LLM systems.

It focuses on understanding:

* how retrieval works
* how embeddings are used
* how LLMs generate grounded answers

instead of treating LLMs as black-box APIs.

---
