"""
Configuration file for RAG Lab
Students: Set your API keys in the .env file (recommended) or environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Langfuse Configuration (for hallucination monitoring)
# Option 1: Set environment variables (recommended for security)
#   Windows PowerShell: $env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
#   Linux/Mac: export LANGFUSE_PUBLIC_KEY="pk-lf-..."
# Option 2: Set directly here (easier, but less secure - don't commit to git!)
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")  # Replace "" with "pk-lf-your-key" to set directly
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")  # Replace "" with "sk-lf-your-key" to set directly
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Optional: use local instance

# Hallucination Detection Configuration
HALLUCINATION_SIMILARITY_THRESHOLD = 0.3  # Below this = likely hallucination
HALLUCINATION_FACT_CHECK_ENABLED = True   # Enable fact-checking (requires API calls)
HALLUCINATION_CROSS_VAL_ENABLED = False   # Enable cross-validation (slower, more API calls)
HALLUCINATION_DETECTION_METHODS = ['similarity', 'fact_checking']  # Methods to use