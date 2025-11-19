"""
Run this script to generate all RAG lab files
Usage: python generate_all_files.py
"""

import os

def create_file(filename, content):
    """Create a file with given content"""
    # Create directory if needed
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ Created {filename}")

# ============= CONFIG.PY =============
config_py = '''"""
Configuration file for RAG Lab
Students: Set your Gemini API key in environment variables
"""

import os

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-pro"

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Relevance Scoring
SIMILARITY_THRESHOLD = 0.5
TOP_K_RESULTS = 5

# Guardrails Configuration
PII_PATTERNS = {
    'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
    'phone': r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b',
    'ssn': r'\\b\\d{3}-\\d{2}-\\d{4}\\b',
    'credit_card': r'\\b\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4}\\b'
}

TOXICITY_THRESHOLD = 0.7

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = "student_documents"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
'''

# ============= DOCUMENT_PROCESSOR.PY =============
document_processor_py = '''"""
Assignment 1: Document Ingestion Pipeline
Students: Complete the TODO sections to implement document processing
"""

import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
from config import *


class DocumentProcessor:
    """Handles document ingestion, chunking, and embedding"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "Student RAG documents"}
        )
    
    def clean_text(self, text: str) -> str:
        """
        TODO: Implement text cleaning for embedding hygiene
        
        Tasks:
        1. Remove extra whitespace (multiple spaces, tabs, newlines)
        2. Normalize unicode characters
        3. Remove special characters that don't add meaning
        4. Convert to lowercase (optional - discuss pros/cons)
        
        Args:
            text: Raw text from document
            
        Returns:
            Cleaned text ready for embedding
        """
        # HINT: Use regex and string methods
        # Example: text = re.sub(r'\\s+', ' ', text)
        
        # YOUR CODE HERE
        cleaned = text
        
        return cleaned.strip()
    
    def chunk_document(self, text: str, chunk_size: int = CHUNK_SIZE, 
                      overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        """
        TODO: Split document into overlapping chunks
        
        Tasks:
        1. Split text into chunks of approximately chunk_size characters
        2. Create overlap between chunks to preserve context
        3. Add metadata (chunk_id, position, etc.)
        
        Args:
            text: Cleaned document text
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dicts with 'text', 'chunk_id', 'position' keys
        """
        chunks = []
        
        # YOUR CODE HERE
        # HINT: Use string slicing with a sliding window
        # Example structure:
        # for i in range(0, len(text), chunk_size - overlap):
        #     chunk_text = text[i:i + chunk_size]
        #     chunks.append({...})
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """
        TODO: Generate embeddings for text chunks
        
        Tasks:
        1. Extract text from chunk dictionaries
        2. Use self.embedding_model to encode texts
        3. Convert to list format for ChromaDB
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of embedding vectors
        """
        # YOUR CODE HERE
        # HINT: self.embedding_model.encode() returns numpy arrays
        # Convert to list: embeddings.tolist()
        
        texts = []  # Extract texts from chunks
        embeddings = []  # Generate embeddings
        
        return embeddings
    
    def ingest_document(self, document_text: str, document_name: str) -> int:
        """
        Complete pipeline: clean, chunk, embed, and store document
        
        Args:
            document_text: Raw document text
            document_name: Name/identifier for the document
            
        Returns:
            Number of chunks created
        """
        print(f"Processing document: {document_name}")
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(document_text)
        print(f"✓ Text cleaned ({len(cleaned_text)} characters)")
        
        # Step 2: Chunk document
        chunks = self.chunk_document(cleaned_text)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        embeddings = self.create_embeddings(chunks)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in ChromaDB
        if embeddings and chunks:
            ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'source': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'position': chunk['position']
                }
                for chunk in chunks
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✓ Stored in ChromaDB")
        
        return len(chunks)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': CHROMA_COLLECTION_NAME
        }


# Test your implementation
if __name__ == "__main__":
    # Create sample document
    sample_text = """
    Artificial Intelligence has revolutionized many fields. Machine learning, 
    a subset of AI, enables computers to learn from data without being explicitly 
    programmed. Deep learning, using neural networks with multiple layers, has 
    achieved remarkable results in image recognition, natural language processing, 
    and game playing. The future of AI promises even more exciting developments.
    """
    
    processor = DocumentProcessor()
    
    # Test cleaning
    print("Testing text cleaning...")
    cleaned = processor.clean_text(sample_text)
    print(f"Original length: {len(sample_text)}, Cleaned length: {len(cleaned)}")
    
    # Test chunking
    print("\\nTesting chunking...")
    chunks = processor.chunk_document(cleaned, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2
        print(f"Chunk {i}: {chunk['text'][:50]}...")
    
    # Test full ingestion
    print("\\nTesting full ingestion...")
    num_chunks = processor.ingest_document(sample_text, "sample_doc")
    print(f"\\nSuccess! Ingested {num_chunks} chunks")
    
    # Show stats
    stats = processor.get_collection_stats()
    print(f"\\nCollection stats: {stats}")
'''

# ============= RELEVANCE_SCORER.PY =============
relevance_scorer_py = '''"""
Assignment 2: Relevance Scoring & Ranking
Students: Complete the TODO sections to implement scoring methods
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from config import *


class RelevanceScorer:
    """Handles relevance scoring and result ranking"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        TODO: Calculate cosine similarity between two vectors
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Tasks:
        1. Compute dot product of vectors
        2. Compute magnitudes (norms) of each vector
        3. Return cosine similarity score (0 to 1)
        
        Args:
            vec1, vec2: Embedding vectors
            
        Returns:
            Similarity score between 0 and 1
        """
        # YOUR CODE HERE
        # HINT: Use numpy for easier computation
        # a = np.array(vec1)
        # b = np.array(vec2)
        # similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarity = 0.0
        return similarity
    
    def keyword_overlap_score(self, query: str, document: str) -> float:
        """
        TODO: Calculate keyword overlap between query and document
        
        Tasks:
        1. Tokenize query and document (split into words)
        2. Convert to sets (remove duplicates)
        3. Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        
        Args:
            query: User's question
            document: Retrieved document text
            
        Returns:
            Overlap score between 0 and 1
        """
        # YOUR CODE HERE
        # HINT: Use set operations
        # query_words = set(query.lower().split())
        # doc_words = set(document.lower().split())
        # intersection = query_words & doc_words
        # union = query_words | doc_words
        
        overlap = 0.0
        return overlap
    
    def calculate_relevance_scores(self, query: str, results: Dict) -> List[Dict]:
        """
        TODO: Calculate multiple relevance metrics for search results
        
        Tasks:
        1. Get query embedding
        2. For each result, calculate:
           - Cosine similarity (already done by ChromaDB as 'distance')
           - Keyword overlap score
           - Combined score (weighted average)
        3. Add scores to result dictionaries
        
        Args:
            query: User's search query
            results: Raw results from ChromaDB query
            
        Returns:
            Enhanced results with multiple scores
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        scored_results = []
        
        # YOUR CODE HERE
        # HINT: results contains 'documents', 'distances', 'metadatas'
        # Loop through results and add scores:
        # for i, doc in enumerate(results['documents'][0]):
        #     cosine_sim = 1 - results['distances'][0][i]  # ChromaDB returns distance
        #     keyword_score = self.keyword_overlap_score(query, doc)
        #     combined = 0.7 * cosine_sim + 0.3 * keyword_score
        #     scored_results.append({...})
        
        return scored_results
    
    def filter_by_threshold(self, results: List[Dict], 
                           threshold: float = SIMILARITY_THRESHOLD,
                           score_key: str = 'combined_score') -> List[Dict]:
        """
        TODO: Filter results below relevance threshold
        
        Tasks:
        1. Filter results where score >= threshold
        2. Sort by score (highest first)
        
        Args:
            results: Scored results
            threshold: Minimum score to keep
            score_key: Which score to use for filtering
            
        Returns:
            Filtered and sorted results
        """
        # YOUR CODE HERE
        # HINT: Use list comprehension and sorted()
        # filtered = [r for r in results if r[score_key] >= threshold]
        # return sorted(filtered, key=lambda x: x[score_key], reverse=True)
        
        filtered = results
        return filtered
    
    def search_and_score(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Complete pipeline: search, score, filter, and rank results
        
        Args:
            query: User's search query
            n_results: Number of results to retrieve initially
            
        Returns:
            Filtered and scored results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Calculate relevance scores
        scored_results = self.calculate_relevance_scores(query, results)
        
        # Filter by threshold
        filtered_results = self.filter_by_threshold(scored_results)
        
        return filtered_results
    
    def print_results(self, results: List[Dict], show_scores: bool = True):
        """Pretty print search results"""
        print(f"\\nFound {len(results)} relevant results:\\n")
        
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Text: {result['text'][:100]}...")
            
            if show_scores:
                print(f"Scores:")
                print(f"  - Cosine Similarity: {result.get('cosine_similarity', 0):.3f}")
                print(f"  - Keyword Overlap: {result.get('keyword_overlap', 0):.3f}")
                print(f"  - Combined Score: {result.get('combined_score', 0):.3f}")
            
            print(f"Source: {result.get('source', 'unknown')}")
            print("-" * 80)


# Test your implementation
if __name__ == "__main__":
    scorer = RelevanceScorer()
    
    # Test cosine similarity
    print("Testing cosine similarity...")
    vec1 = [1, 0, 0]
    vec2 = [1, 0, 0]
    vec3 = [0, 1, 0]
    
    sim_same = scorer.cosine_similarity(vec1, vec2)
    sim_diff = scorer.cosine_similarity(vec1, vec3)
    print(f"Same vectors: {sim_same} (should be ~1.0)")
    print(f"Orthogonal vectors: {sim_diff} (should be ~0.0)")
    
    # Test keyword overlap
    print("\\nTesting keyword overlap...")
    query = "machine learning algorithms"
    doc1 = "Machine learning uses various algorithms for pattern recognition"
    doc2 = "The weather today is sunny and warm"
    
    overlap1 = scorer.keyword_overlap_score(query, doc1)
    overlap2 = scorer.keyword_overlap_score(query, doc2)
    print(f"Relevant doc overlap: {overlap1}")
    print(f"Irrelevant doc overlap: {overlap2}")
    
    # Test full search (requires ingested documents)
    print("\\nTesting search and scoring...")
    if scorer.collection.count() > 0:
        results = scorer.search_and_score("artificial intelligence")
        scorer.print_results(results)
    else:
        print("No documents in collection. Run document_processor.py first!")
'''

# ============= GUARDRAILS.PY =============
guardrails_py = '''"""
Assignment 3: Guardrails System
Students: Complete the TODO sections to implement safety checks
"""

import re
from typing import Dict, List, Tuple
from transformers import pipeline
from config import *


class Guardrails:
    """Handles content safety: PII detection and toxicity checking"""
    
    def __init__(self):
        """
        Initialize guardrails with toxicity detection model
        Note: First run will download the model (~500MB)
        """
        try:
            self.toxicity_detector = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                top_k=None
            )
            self.toxicity_enabled = True
        except Exception as e:
            print(f"Warning: Could not load toxicity model: {e}")
            print("Toxicity detection will be disabled.")
            self.toxicity_enabled = False
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        TODO: Detect personally identifiable information in text
        
        Tasks:
        1. Use regex patterns from config.py to find PII
        2. Return dictionary with PII types and found instances
        3. Handle empty/no matches gracefully
        
        Args:
            text: Text to scan for PII
            
        Returns:
            Dict mapping PII type to list of found instances
            Example: {'email': ['test@example.com'], 'phone': ['555-1234']}
        """
        found_pii = {}
        
        # YOUR CODE HERE
        # HINT: Loop through PII_PATTERNS from config
        # for pii_type, pattern in PII_PATTERNS.items():
        #     matches = re.findall(pattern, text)
        #     if matches:
        #         found_pii[pii_type] = matches
        
        return found_pii
    
    def check_toxicity(self, text: str) -> Tuple[bool, float, Dict]:
        """
        TODO: Check text for toxic content using ML model
        
        Tasks:
        1. Use self.toxicity_detector to analyze text
        2. Extract toxicity score from results
        3. Compare against TOXICITY_THRESHOLD
        4. Return (is_toxic, score, details)
        
        Args:
            text: Text to check for toxicity
            
        Returns:
            Tuple of (is_toxic: bool, max_score: float, details: dict)
        """
        if not self.toxicity_enabled:
            return False, 0.0, {"note": "Toxicity detection disabled"}
        
        if not text or len(text.strip()) == 0:
            return False, 0.0, {"note": "Empty text"}
        
        # YOUR CODE HERE
        # HINT: self.toxicity_detector returns list of dicts with 'label' and 'score'
        # results = self.toxicity_detector(text[:512])  # Limit to 512 chars
        # Find 'toxic' label and check if score > TOXICITY_THRESHOLD
        
        is_toxic = False
        max_score = 0.0
        details = {}
        
        return is_toxic, max_score, details
    
    def validate_query(self, query: str) -> Dict:
        """
        TODO: Complete validation check for user queries
        
        Tasks:
        1. Check for PII in query
        2. Check for toxicity in query
        3. Return comprehensive validation result
        
        Args:
            query: User's input query
            
        Returns:
            Dict with 'safe', 'warnings', 'pii_found', 'toxicity' keys
        """
        result = {
            'safe': True,
            'warnings': [],
            'pii_found': {},
            'toxicity': {
                'is_toxic': False,
                'score': 0.0
            }
        }
        
        # YOUR CODE HERE
        # 1. Check for PII
        # pii = self.detect_pii(query)
        # if pii:
        #     result['safe'] = False
        #     result['warnings'].append("PII detected in query")
        #     result['pii_found'] = pii
        
        # 2. Check toxicity
        # is_toxic, score, details = self.check_toxicity(query)
        # if is_toxic:
        #     result['safe'] = False
        #     result['warnings'].append("Toxic content detected")
        #     result['toxicity'] = {'is_toxic': True, 'score': score, 'details': details}
        
        return result
    
    def validate_response(self, response: str) -> Dict:
        """
        Validate generated response before showing to user
        
        Args:
            response: AI-generated response text
            
        Returns:
            Validation result dict
        """
        # Same logic as validate_query
        return self.validate_query(response)
    
    def sanitize_text(self, text: str) -> str:
        """
        TODO: Remove or mask PII from text
        
        Tasks:
        1. Find PII using detect_pii()
        2. Replace found PII with masked versions
        3. Return sanitized text
        
        Args:
            text: Text potentially containing PII
            
        Returns:
            Text with PII masked/removed
        """
        sanitized = text
        
        # YOUR CODE HERE
        # HINT: For each PII type, replace matches with [REDACTED_TYPE]
        # pii = self.detect_pii(text)
        # for pii_type, instances in pii.items():
        #     for instance in instances:
        #         sanitized = sanitized.replace(instance, f"[REDACTED_{pii_type.upper()}]")
        
        return sanitized
    
    def print_validation_result(self, result: Dict, text_preview: str = ""):
        """Pretty print validation results"""
        print("\\n" + "="*80)
        print("GUARDRAILS VALIDATION RESULT")
        print("="*80)
        
        if text_preview:
            print(f"\\nText preview: {text_preview[:100]}...")
        
        status = "✓ SAFE" if result['safe'] else "⚠ FLAGGED"
        print(f"\\nStatus: {status}")
        
        if result['warnings']:
            print(f"\\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        if result['pii_found']:
            print(f"\\nPII Detected:")
            for pii_type, instances in result['pii_found'].items():
                print(f"  - {pii_type}: {len(instances)} instance(s)")
                for instance in instances[:3]:  # Show first 3
                    print(f"    * {instance}")
        
        if result['toxicity']['is_toxic']:
            print(f"\\nToxicity:")
            print(f"  - Score: {result['toxicity']['score']:.3f}")
            print(f"  - Threshold: {TOXICITY_THRESHOLD}")
        
        print("="*80 + "\\n")


# Test your implementation
if __name__ == "__main__":
    print("Initializing Guardrails (may download model on first run)...")
    guardrails = Guardrails()
    
    # Test PII detection
    print("\\n" + "="*80)
    print("TEST 1: PII Detection")
    print("="*80)
    
    test_texts = [
        "My email is john.doe@example.com and phone is 555-123-4567",
        "Contact me at alice@company.org or call 123-456-7890",
        "SSN: 123-45-6789, Card: 4532-1234-5678-9010",
        "This text has no PII at all"
    ]
    
    for text in test_texts:
        print(f"\\nText: {text}")
        pii = guardrails.detect_pii(text)
        if pii:
            print(f"Found PII: {pii}")
            sanitized = guardrails.sanitize_text(text)
            print(f"Sanitized: {sanitized}")
        else:
            print("No PII detected")
    
    # Test toxicity detection
    print("\\n" + "="*80)
    print("TEST 2: Toxicity Detection")
    print("="*80)
    
    if guardrails.toxicity_enabled:
        toxic_texts = [
            "This is a friendly and helpful message",
            "You are stupid and worthless",
            "I disagree with your opinion, but respect your view"
        ]
        
        for text in toxic_texts:
            print(f"\\nText: {text}")
            is_toxic, score, details = guardrails.check_toxicity(text)
            print(f"Toxic: {is_toxic}, Score: {score:.3f}")
    else:
        print("Toxicity detection disabled")
    
    # Test full query validation
    print("\\n" + "="*80)
    print("TEST 3: Full Query Validation")
    print("="*80)
    
    test_queries = [
        "What is machine learning?",
        "My email is test@example.com, can you help?",
        "You're an idiot, answer my question!"
    ]
    
    for query in test_queries:
        result = guardrails.validate_query(query)
        guardrails.print_validation_result(result, query)
'''

# ============= RAG_APP.PY =============
rag_app_py = '''"""
Assignment 4: Complete Streamlit RAG Interface
Students: Complete the TODO sections to build the full application
"""

import streamlit as st
import google.generativeai as genai
from document_processor import DocumentProcessor
from relevance_scorer import RelevanceScorer
from guardrails import Guardrails
from config import *

# Configure page
st.set_page_config(
    page_title="RAG Document Query System",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all RAG components (cached for performance)"""
    doc_processor = DocumentProcessor()
    scorer = RelevanceScorer()
    guardrails = Guardrails()
    
    # Configure Gemini
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
    else:
        model = None
    
    return doc_processor, scorer, guardrails, model


def generate_rag_response(model, query: str, context_chunks: list) -> str:
    """
    TODO: Generate response using Gemini with retrieved context
    
    Tasks:
    1. Format context chunks into a clear context section
    2. Create a prompt that includes context and query
    3. Call Gemini API to generate response
    4. Handle errors gracefully
    
    Args:
        model: Gemini model instance
        query: User's question
        context_chunks: List of relevant document chunks
        
    Returns:
        Generated response text
    """
    if not model:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    # YOUR CODE HERE
    # HINT: Format like this:
    # context = "\\n\\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
    # prompt = f"""Based on the following context, answer the question...
    # Context:
    # {context}
    # Question: {query}
    # Answer:"""
    # response = model.generate_content(prompt)
    # return response.text
    
    response = "TODO: Implement RAG response generation"
    return response


def main():
    # Initialize components
    doc_processor, scorer, guardrails, gemini_model = initialize_components()
    
    # Title and description
    st.title("📚 RAG Document Query System")
    st.markdown("""
    Upload a document, then ask questions about its content. 
    The system uses RAG (Retrieval-Augmented Generation) with guardrails.
    """)
    
    # Sidebar: Document Management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['txt'],
            help="Upload a text document to query"
        )
        
        if uploaded_file:
            # TODO: Process uploaded file
            # HINT:
            # 1. Read file content: text = uploaded_file.read().decode('utf-8')
            # 2. Call doc_processor.ingest_document(text, uploaded_file.name)
            # 3. Show success message with st.success()
            # 4. Show stats with doc_processor.get_collection_stats()
            
            st.info("TODO: Implement document upload processing")
        
        # Show collection stats
        st.subheader("📊 Collection Stats")
        stats = doc_processor.get_collection_stats()
        st.metric("Total Chunks", stats['total_chunks'])
        st.caption(f"Collection: {stats['collection_name']}")
        
        # Configuration expander
        with st.expander("⚙️ Configuration"):
            st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
            st.write(f"**Chunk Size:** {CHUNK_SIZE}")
            st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
            st.write(f"**Similarity Threshold:** {SIMILARITY_THRESHOLD}")
            st.write(f"**Toxicity Threshold:** {TOXICITY_THRESHOLD}")
    
    # Main query interface
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        help="Ask anything about the uploaded document"
    )
    
    # Search button
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        show_details = st.checkbox("Show detailed scores", value=True)
    
    # Process query when button clicked
    if search_button and query:
        # TODO: Implement complete query processing pipeline
        
        # Step 1: Validate query with guardrails
        st.subheader("🛡️ Guardrails Check")
        with st.spinner("Checking query safety..."):
            # YOUR CODE HERE
            # validation = guardrails.validate_query(query)
            # Display validation results
            pass
        
        st.info("TODO: Implement guardrails validation")
        
        # Step 2: Search for relevant chunks
        st.subheader("🔍 Document Search")
        with st.spinner("Searching relevant content..."):
            # YOUR CODE HERE
            # results = scorer.search_and_score(query)
            # Display number of results found
            pass
        
        st.info("TODO: Implement document search")
        
        # Step 3: Display search results
        if show_details:
            st.subheader("📄 Retrieved Chunks")
            # YOUR CODE HERE
            # Loop through results and display in expandable sections
            # for i, result in enumerate(results):
            #     with st.expander(f"Chunk {i+1} - Score: {result['combined_score']:.3f}"):
            #         st.write(result['text'])
            #         st.caption(f"Source: {result['source']}")
            pass
        
        # Step 4: Generate RAG response
        st.subheader("🤖 AI Response")
        with st.spinner("Generating response..."):
            # YOUR CODE HERE
            # response = generate_rag_response(gemini_model, query, results)
            # Display response
            pass
        
        st.info("TODO: Implement RAG response generation")
        
        # Step 5: Validate response with guardrails
        st.subheader("✅ Response Validation")
        with st.spinner("Validating response..."):
            # YOUR CODE HERE
            # response_validation = guardrails.validate_response(response)
            # Display validation status
            pass
        
        st.info("TODO: Implement response validation")
    
    elif search_button and not query:
        st.warning("Please enter a question first!")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. **Upload a document** in the sidebar (text file)
    2. **Enter a question** about the document
    3. **Click Search** to get AI-powered answers
    4. The system will check for safety, find relevant content, and generate a response
    
    ### 🛡️ Safety Features:
    - **PII Detection**: Identifies emails, phone numbers, SSNs
    - **Toxicity Check**: Filters inappropriate content
    - **Relevance Scoring**: Ensures accurate answers
    """)


if __name__ == "__main__":
    main()
'''

# ============= REQUIREMENTS.TXT =============
requirements_txt = '''streamlit==1.31.0
chromadb==0.4.22
sentence-transformers==2.3.1
google-generativeai==0.3.2
transformers==4.37.2
torch==2.2.0
numpy==1.26.3
'''

# ============= SAMPLE DOCUMENT =============
sample_doc = '''The History of Artificial Intelligence

Artificial Intelligence (AI) has a rich history spanning over seven decades. The field officially began in 1956 at the Dartmouth Conference, where John McCarthy coined the term "artificial intelligence." Early pioneers including Alan Turing, Marvin Minsky, and Herbert Simon laid the groundwork for what would become one of the most transformative technologies of our time.

The Early Years (1950s-1960s)
The 1950s saw the birth of AI as a formal academic discipline. Alan Turing proposed the famous Turing Test in 1950 as a measure of machine intelligence. The test asks whether a machine can exhibit intelligent behavior indistinguishable from a human. In 1956, the Logic Theorist program, created by Allen Newell and Herbert Simon, became one of the first AI programs capable of proving mathematical theorems.

During this period, researchers were optimistic about AI's potential. Many believed that machines with human-level intelligence were just around the corner. Early programs showed promising results in game playing, particularly checkers and chess, and simple problem-solving tasks.

The First AI Winter (1970s-1980s)
By the 1970s, the initial optimism began to fade. The limitations of early AI systems became apparent. Computers lacked the processing power and memory needed for complex tasks. Furthermore, many problems proved more difficult than initially anticipated. The British government's Lighthill Report in 1973 criticized AI research for failing to achieve its ambitious goals, leading to significant funding cuts.

This period, known as the "AI Winter," saw reduced investment and interest in AI research. However, some important developments still occurred. Expert systems, which could mimic human decision-making in specific domains, gained traction in the 1980s. Companies began using these systems for tasks like medical diagnosis and mineral exploration.

The Rise of Machine Learning (1990s-2000s)
The 1990s marked a turning point for AI. Researchers shifted focus from hand-coded rules to machine learning approaches. Instead of programming explicit instructions, systems could learn patterns from data. This paradigm shift proved revolutionary.

In 1997, IBM's Deep Blue defeated world chess champion Garry Kasparov, demonstrating that machines could excel at tasks requiring strategic thinking. The development of the internet provided vast amounts of data for training machine learning models. Statistical methods and neural networks, which had been largely abandoned during the AI winter, experienced a renaissance.

The Deep Learning Revolution (2010s)
The 2010s witnessed an explosion in AI capabilities, driven primarily by deep learning. Deep neural networks, inspired by the structure of the human brain, achieved unprecedented performance on complex tasks. Three key factors enabled this revolution: abundant data, powerful GPUs for parallel computation, and algorithmic innovations.

In 2012, AlexNet won the ImageNet competition by a large margin, using deep convolutional neural networks. This breakthrough demonstrated that deep learning could dramatically outperform traditional computer vision methods. Subsequently, deep learning transformed natural language processing, speech recognition, and many other domains.

Modern AI and Future Directions
Today, AI pervades everyday life. Virtual assistants like Siri and Alexa use natural language processing. Social media platforms employ AI for content recommendation. Autonomous vehicles navigate streets using computer vision and deep learning. Medical AI systems assist in diagnosis and drug discovery.

Large language models, trained on vast text corpora, can generate human-like text, answer questions, and assist with coding. These models represent a significant leap in AI capabilities. However, they also raise important questions about AI safety, bias, and societal impact.

Current research focuses on several frontiers: making AI systems more efficient, improving interpretability of AI decisions, developing AI that can reason more like humans, creating artificial general intelligence that matches human-level intelligence across all domains, and ensuring AI alignment with human values and ethical principles.
'''

# Create all files
print("Creating RAG Lab files...\n")

create_file("config.py", config_py)
create_file("document_processor.py", document_processor_py)
create_file("relevance_scorer.py", relevance_scorer_py)
create_file("guardrails.py", guardrails_py)
create_file("rag_app.py", rag_app_py)
create_file("requirements.txt", requirements_txt)
create_file("sample_documents/ai_history.txt", sample_doc)

print("\n" + "="*80)
print("✅ All files created successfully!")
print("="*80)
print("\nNext steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Set Gemini API key: export GEMINI_API_KEY=your_key_here")
print("3. Test each assignment file individually")
print("4. Run the Streamlit app: streamlit run rag_app.py")
print("\nFor detailed instructions, check the documentation files.")
print("="*80)