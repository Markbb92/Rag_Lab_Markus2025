"""
Configuration file for RAG Lab
Students: Set your Gemini API key in environment variables
"""

import os

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Relevance Scoring
SIMILARITY_THRESHOLD = 0.1
TOP_K_RESULTS = 5

# Guardrails Configuration
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
}

TOXICITY_THRESHOLD = 0.7

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "student_documents"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
