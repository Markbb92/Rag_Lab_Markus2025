"""
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
        Implement text cleaning for embedding hygiene
        
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
        # Example: text = re.sub(r'\s+', ' ', text)
        
        # YOUR CODE HERE
        # Step 1: Remove extra whitespace (multiple spaces, tabs, newlines)
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Step 2: Normalize unicode characters (handle accented characters, etc.)
        # This converts things like é to e, ñ to n, etc.
        import unicodedata
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Step 3: Remove special characters that don't add meaning
        # Keep letters, numbers, spaces, and basic punctuation
        cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', cleaned)
        
        # Step 4: Convert to lowercase (helps with consistency)
        # Pro: Better matching, case-insensitive search
        # Con: Loses emphasis, proper nouns might be less clear
        cleaned = cleaned.lower()
        
        return cleaned.strip()
    
    def chunk_document(self, text: str, chunk_size: int = CHUNK_SIZE, 
                      overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        """
        Split document into overlapping chunks
        
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
        
        # Handle edge case: if text is shorter than chunk_size, return single chunk
        if len(text) <= chunk_size:
            return [{
                'text': text,
                'chunk_id': 0,
                'position': 0
            }]
        
        # Calculate step size (how much to move forward each iteration)
        step_size = chunk_size - overlap
        
        # Split text into overlapping chunks
        for i in range(0, len(text), step_size):
            # Get chunk text
            chunk_text = text[i:i + chunk_size]
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            # Create chunk dictionary with metadata
            chunk_dict = {
                'text': chunk_text,
                'chunk_id': len(chunks),
                'position': i
            }
            
            chunks.append(chunk_dict)
            
            # If we've reached the end of the text, break
            if i + chunk_size >= len(text):
                break
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """
        Generate embeddings for text chunks
        
        Tasks:
        1. Extract text from chunk dictionaries
        2. Use self.embedding_model to encode texts
        3. Convert to list format for ChromaDB
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings using the sentence transformer model
        embeddings = self.embedding_model.encode(texts)
        
        # Convert numpy arrays to list format for ChromaDB
        embeddings_list = embeddings.tolist()
        
        return embeddings_list
    
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
    print("\nTesting chunking...")
    chunks = processor.chunk_document(cleaned, chunk_size=100, overlap=20)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:2]):  # Show first 2
        print(f"Chunk {i}: {chunk['text'][:50]}...")
    
    # Test full ingestion
    print("\nTesting full ingestion...")
    num_chunks = processor.ingest_document(sample_text, "sample_doc")
    print(f"\nSuccess! Ingested {num_chunks} chunks")
    
    # Show stats
    stats = processor.get_collection_stats()
    print(f"\nCollection stats: {stats}")
