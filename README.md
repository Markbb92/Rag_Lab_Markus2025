# 📚 Advanced RAG Chatbot with Safety Guardrails

A sophisticated Retrieval-Augmented Generation (RAG) system built with **Gemini 2.5 Flash**, **ChromaDB**, and **Streamlit**. Unlike basic RAG setups, this implementation includes a multi-layer safety pipeline, custom relevance scoring, and hallucination monitoring.

## 🚀 Key Features

* **Intelligent Ingestion**: Processes PDFs, DOCX, and TXT files with automated cleaning and semantic chunking.
* **Safety Guardrails**: Integrated `toxic-bert` model and regex-based PII detection to filter sensitive or harmful data before and after generation.
* **Hybrid Relevance Scorer**: Custom logic combining Cosine Similarity and Keyword Overlap to ensure only the most relevant context reaches the LLM.
* **Hallucination Monitor**: Real-time response validation using semantic similarity and fact-checking to ensure AI honesty.
* **Observability**: Integrated with **Langfuse** for tracing and performance metrics.

## 🏗️ System Architecture

![Architecture Diagram](./architecture.png)
> *Note: If you haven't uploaded your diagram yet, see the "Architecture" section in the docs.*

## 🛠️ Tech Stack

- **LLM:** Google Gemini 2.5 Flash
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Frontend:** Streamlit
- **Monitoring:** Langfuse
- **Safety:** Transformers (Toxic-BERT)

## 📋 Quick Start

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
