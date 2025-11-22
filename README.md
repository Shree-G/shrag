---
title: Shrag
emoji: 🐢
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
short_description: rag for shree's personal website
---

# shrag: Personal RAG Chatbot API

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-teal)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

**shrag** (Shree's RAG) is a production-grade, full-stack Retrieval-Augmented Generation (RAG) API designed to answer questions about my professional background, projects, and skills.

Unlike simple tutorial bots, this project focuses on **engineering rigor**: it features a decoupled architecture, a formal evaluation pipeline, semantic chunking strategies, and containerized deployment.

## 🚀 Key Features

* **Production Architecture:** Built as a standalone **FastAPI** service, decoupled from the frontend to allow for independent scaling and flexible UI integration (currently serving an `assistant-ui` frontend).
* **Advanced Ingestion Pipeline:** Uses **Semantic Chunking** (via `langchain-experimental`) rather than naive character splitting to ensure retrieved context maintains semantic integrity.
* **Conversational Memory:** Implements a history-aware retrieval chain that re-phrases follow-up questions to maintain context across multi-turn conversations.
* **Rigorous Evaluation:** The pipeline is not just "eye-balled." It is formally evaluated using **DeepEval** and **LangSmith** against a synthetic "Golden Dataset" to measure Faithfulness, Answer Relevancy, and Contextual Precision.
* **Containerized Deployment:** Fully Dockerized and deployed via **Hugging Face Spaces**, ensuring a consistent environment from development to production.

## 🛠️ Tech Stack

### Core AI & RAG
* **Orchestration:** LangChain (Python)
* **LLM Inference:** [Groq](https://groq.com/) (Llama 3.3 70B & Llama 3.1 8B) for sub-second latency.
* **Vector Database:** ChromaDB (Persistent local storage).
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`).
* **Ingestion:** Custom Python scripts using `SemanticChunker`.

### Backend Engineering
* **Framework:** FastAPI (Asynchronous Python web server).
* **Validation:** Pydantic (Strict data contracts for Requests/Responses).
* **Server:** Uvicorn.

### DevOps & Evaluation
* **Containerization:** Docker (Multi-stage builds).
* **Hosting:** Hugging Face Spaces (Docker Space).
* **Observability:** LangSmith (Tracing and debugging).
* **Evaluation Metrics:** DeepEval (Unit testing for RAG performance).

## 🧠 Architecture Overview

### 1. The Data Pipeline (`/scripts/ingest.py`)
The system ingests data from multiple sources:
* **PDF Resume:** Parsed and chunked.
* **GitHub Repositories:** Dynamically fetches `README.md` files from my public GitHub profile using the GitHub API.
* **Personal Data:** Ingests structured CSV data regarding coursework and skills.

**Chunking Strategy:** I moved away from naive `RecursiveCharacterTextSplitter` to **Semantic Chunking**. This calculates the cosine similarity between sentences and only "breaks" a chunk when the topic shifts, resulting in higher quality context for the LLM.

### 2. The RAG Chain (`/core/chain.py`)
The application uses a **Conversational Retrieval Chain** pattern:
1.  **Rephrase:** A lightweight LLM (Llama 3 8B) takes the user's query and chat history to generate a standalone search query.
2.  **Retrieve:** This query searches the ChromaDB vector store for the top `k` (tuned to k=3) semantically relevant chunks.
3.  **Generate:** The retrieved context and the original question are passed to the synthesis LLM (Llama 3 70B or 8B) to generate the final response.

### 3. The API Layer (`/app/api.py`)
The chain is exposed via a RESTful `/chat` endpoint. It handles:
* **Session Management:** Tracking conversation history via `session_id`.
* **CORS:** Allowing secure requests from the Next.js frontend.
* **Source Citation:** Returning metadata about which files (e.g., `Resume.pdf`, `shrocial_media.git`) were used to generate the answer.

## 🧪 Evaluation & Testing

To ensure reliability, I implemented an automated evaluation pipeline (`evaluate.py`) that integrates **DeepEval** with **LangSmith**.

* **Golden Dataset:** A curated set of 40+ QA pairs covering factual recall, summarization, and negative constraints (questions the bot *shouldn't* answer).
* **Metrics Tracked:**
    * **Faithfulness:** Does the answer come *only* from the context? (Prevents hallucinations).
    * **Answer Relevancy:** Did the bot actually answer the user's question?
    * **Contextual Precision:** Did the retriever find the right document?

## 📦 Installation & Setup

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/Shree-G/shrag.git](https://github.com/Shree-G/shrag.git)
    cd shrag
    ```

2.  **Set up Environment:**
    Create a `.env` file with the following keys:
    ```env
    GROQ_API_KEY=gsk_...
    LANGCHAIN_API_KEY=lsv2_...
    OPENAI_API_KEY=sk-... (Only used for Evaluation/Judging)
    ```

3.  **Install Dependencies & Build Database:**
    ```bash
    # Install libraries
    pip install -r requirements.txt

    # Create the local ChromaDB
    python scripts/ingest.py
    ```

4.  **Run via Docker:**
    ```bash
    docker build -t shrag-api .
    docker run -p 8000:8000 --env-file .env shrag-api
    ```

5.  **Access API:**
    Open `http://localhost:8000/docs` to see the Swagger UI.