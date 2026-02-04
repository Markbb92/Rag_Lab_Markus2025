"""
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
        Calculate cosine similarity between two vectors
        
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
        # Convert to numpy arrays for easier computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(a, b)
        
        # Calculate magnitudes (norms)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (norm_a * norm_b)
        
        return similarity
    
    def keyword_overlap_score(self, query: str, document: str) -> float:
        """
        Calculate keyword overlap between query and document
        
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
        # Tokenize and convert to lowercase
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Calculate intersection (common words)
        intersection = query_words & doc_words
        
        # Calculate union (all unique words)
        union = query_words | doc_words
        
        # Avoid division by zero
        if len(union) == 0:
            return 0.0
        
        # Calculate Jaccard similarity
        overlap = len(intersection) / len(union)
        
        return overlap
    
    def calculate_relevance_scores(self, query: str, results: Dict) -> List[Dict]:
        """
        Calculate multiple relevance metrics for search results
        
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
        
        # Extract results from ChromaDB format
        documents = results['documents'][0]  # First query results
        distances = results['distances'][0]   # ChromaDB distances
        metadatas = results['metadatas'][0]   # Metadata for each result
        
        # Get all document embeddings at once for efficiency
        result_ids = results['ids'][0]
        doc_embeddings_result = self.collection.get(
            ids=result_ids,
            include=['embeddings']
        )
        doc_embeddings = doc_embeddings_result['embeddings'] if doc_embeddings_result['embeddings'] else []
        
        # Calculate scores for each result
        for i, doc in enumerate(documents):
            # Recalculate cosine similarity directly from embeddings
            # This is more accurate than converting ChromaDB distance
            if i < len(doc_embeddings) and doc_embeddings[i]:
                cosine_sim = self.cosine_similarity(query_embedding, doc_embeddings[i])
                # Ensure similarity is in [0, 1] range (should be, but clamp just in case)
                cosine_sim = max(0.0, min(1.0, cosine_sim))
            else:
                # Fallback: try to convert distance (may not work if using L2)
                # ChromaDB cosine distance = 1 - cosine_similarity
                raw_similarity = 1 - distances[i]
                cosine_sim = max(0.0, min(1.0, raw_similarity))
            
            # Calculate keyword overlap score
            keyword_score = self.keyword_overlap_score(query, doc)
            
            # Calculate combined score (weighted average)
            # 70% semantic similarity, 30% keyword overlap
            combined = 0.7 * cosine_sim + 0.3 * keyword_score
            
            # Create enhanced result dictionary
            enhanced_result = {
                'text': doc,
                'cosine_similarity': cosine_sim,
                'keyword_overlap': keyword_score,
                'combined_score': combined,
                'source': metadatas[i].get('source', 'unknown'),
                'chunk_id': metadatas[i].get('chunk_id', i),
                'position': metadatas[i].get('position', 0)
            }
            
            scored_results.append(enhanced_result)
        
        return scored_results
    
    def filter_by_threshold(self, results: List[Dict], 
                           threshold: float = SIMILARITY_THRESHOLD,
                           score_key: str = 'combined_score') -> List[Dict]:
        """
        Filter results below relevance threshold
        
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
        # Filter results above threshold
        filtered = [r for r in results if r[score_key] >= threshold]
        
        # Sort by score (highest first)
        sorted_results = sorted(filtered, key=lambda x: x[score_key], reverse=True)
        
        return sorted_results
    
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
        print(f"\nFound {len(results)} relevant results:\n")
        
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
    print("\nTesting keyword overlap...")
    query = "machine learning algorithms"
    doc1 = "Machine learning uses various algorithms for pattern recognition"
    doc2 = "The weather today is sunny and warm"
    
    overlap1 = scorer.keyword_overlap_score(query, doc1)
    overlap2 = scorer.keyword_overlap_score(query, doc2)
    print(f"Relevant doc overlap: {overlap1}")
    print(f"Irrelevant doc overlap: {overlap2}")
    
    # Test full search (requires ingested documents)
    print("\nTesting search and scoring...")
    if scorer.collection.count() > 0:
        results = scorer.search_and_score("artificial intelligence")
        scorer.print_results(results)
    else:
        print("No documents in collection. Run document_processor.py first!")
