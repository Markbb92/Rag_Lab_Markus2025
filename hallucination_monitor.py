"""
Hallucination Monitoring System
Implements automated detection methods: similarity scoring, fact-checking, and cross-validation
Integrates with Langfuse for observability and baseline metrics
"""

import logging
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import *
from langfuse import Langfuse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hallucination_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HallucinationMonitor:
    """Detects hallucinations in LLM-generated responses"""
    
    def __init__(self, gemini_model=None):
        """
        Initialize hallucination monitor with embedding model and Langfuse
        
        Args:
            gemini_model: Optional Gemini model instance for fact-checking
        """
        # Initialize embedding model for similarity scoring
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize Langfuse client (will use environment variables if not configured)
        try:
            self.langfuse = Langfuse()
            self.langfuse_enabled = True
            logger.info("Langfuse initialized successfully")
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}. Monitoring will continue without Langfuse.")
            self.langfuse_enabled = False
            self.langfuse = None
        
        # Store Gemini model for fact-checking
        self.gemini_model = gemini_model
        
        # Baseline metrics storage
        self.baseline_metrics = {
            'total_queries': 0,
            'hallucination_detected_count': 0,
            'avg_similarity_score': 0.0,
            'avg_fact_check_score': 0.0,
            'similarity_scores': [],
            'fact_check_scores': []
        }
    
    def similarity_scoring(self, response: str, context_chunks: List[Dict]) -> Dict:
        """
        Method 1: Similarity Scoring
        Compare generated response against retrieved context using embeddings
        
        Args:
            response: Generated LLM response
            context_chunks: List of retrieved context chunks
            
        Returns:
            Dict with similarity metrics
        """
        if not context_chunks or not response:
            return {
                'method': 'similarity_scoring',
                'score': 0.0,
                'is_hallucination': True,
                'reason': 'No context or response provided'
            }
        
        # Combine all context chunks
        combined_context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Generate embeddings
        response_embedding = self.embedding_model.encode(response).tolist()
        context_embedding = self.embedding_model.encode(combined_context).tolist()
        
        # Calculate cosine similarity
        from relevance_scorer import RelevanceScorer
        scorer = RelevanceScorer()
        similarity_score = scorer.cosine_similarity(response_embedding, context_embedding)
        
        # Threshold: if similarity < threshold, likely hallucination
        is_hallucination = similarity_score < HALLUCINATION_SIMILARITY_THRESHOLD
        
        result = {
            'method': 'similarity_scoring',
            'score': float(similarity_score),
            'is_hallucination': is_hallucination,
            'threshold': HALLUCINATION_SIMILARITY_THRESHOLD,
            'reason': 'Low semantic similarity to context' if is_hallucination else 'High semantic similarity to context'
        }
        
        logger.info(f"Similarity scoring: score={similarity_score:.3f}, hallucination={is_hallucination}")
        return result
    
    def fact_checking(self, response: str, context_chunks: List[Dict], query: str) -> Dict:
        """
        Method 2: Fact-Checking using LLM
        Use Gemini to verify if response claims are supported by context
        
        Args:
            response: Generated LLM response
            context_chunks: List of retrieved context chunks
            query: Original user query
            
        Returns:
            Dict with fact-checking results
        """
        if not self.gemini_model:
            return {
                'method': 'fact_checking',
                'score': 1.0,
                'is_hallucination': False,
                'reason': 'Gemini model not available for fact-checking'
            }
        
        if not context_chunks or not response:
            return {
                'method': 'fact_checking',
                'score': 0.0,
                'is_hallucination': True,
                'reason': 'No context or response provided'
            }
        
        # Combine context
        combined_context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        # Create fact-checking prompt
        fact_check_prompt = f"""You are a fact-checker. Analyze whether the following response is fully supported by the provided context.

Original Query: {query}

Retrieved Context:
{combined_context}

Generated Response:
{response}

Task: Determine if ALL factual claims in the response are directly supported by the context. 

Respond ONLY with a JSON object in this exact format:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["list of any unsupported claims"],
    "reason": "brief explanation"
}}

If any factual claim in the response cannot be verified in the context, mark supported as false."""

        try:
            fact_check_response = self.gemini_model.generate_content(fact_check_prompt)
            fact_check_text = fact_check_response.text.strip()
            
            # Parse JSON response (handle markdown code blocks)
            import json
            import re
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'\{.*\}', fact_check_text, re.DOTALL)
            if json_match:
                fact_check_json = json.loads(json_match.group())
            else:
                # Try parsing directly
                fact_check_json = json.loads(fact_check_text)
            
            supported = fact_check_json.get('supported', True)
            confidence = fact_check_json.get('confidence', 1.0)
            unsupported_claims = fact_check_json.get('unsupported_claims', [])
            
            # Convert to hallucination score (1.0 = no hallucination, 0.0 = complete hallucination)
            fact_check_score = confidence if supported else (1.0 - confidence)
            is_hallucination = not supported
            
            result = {
                'method': 'fact_checking',
                'score': float(fact_check_score),
                'is_hallucination': is_hallucination,
                'confidence': float(confidence),
                'supported': supported,
                'unsupported_claims': unsupported_claims,
                'reason': fact_check_json.get('reason', 'Fact-check completed')
            }
            
            logger.info(f"Fact-checking: score={fact_check_score:.3f}, hallucination={is_hallucination}")
            return result
            
        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            return {
                'method': 'fact_checking',
                'score': 0.5,  # Neutral score on error
                'is_hallucination': False,
                'error': str(e),
                'reason': 'Fact-checking failed due to error'
            }
    
    def cross_validation(self, response: str, context_chunks: List[Dict], query: str, 
                        num_validations: int = 2) -> Dict:
        """
        Method 3: Cross-Validation (Optional)
        Generate multiple responses and check consistency
        
        Args:
            response: Original generated response
            context_chunks: List of retrieved context chunks
            query: Original user query
            num_validations: Number of additional responses to generate
            
        Returns:
            Dict with cross-validation results
        """
        if not self.gemini_model or num_validations < 1:
            return {
                'method': 'cross_validation',
                'score': 1.0,
                'is_hallucination': False,
                'reason': 'Cross-validation skipped'
            }
        
        # Generate multiple responses
        context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        responses = [response]  # Include original response
        
        for i in range(num_validations):
            try:
                prompt = f"""Based on the following context from uploaded documents, please answer the question accurately and comprehensively.

Context:
{context}

Question: {query}

Instructions:
- Use only the information provided in the context above
- If the context doesn't contain enough information to answer the question, say so
- Be specific and cite relevant parts of the context
- Keep your answer clear and well-structured

Answer:"""
                
                validation_response = self.gemini_model.generate_content(prompt)
                responses.append(validation_response.text)
            except Exception as e:
                logger.warning(f"Cross-validation generation {i+1} failed: {e}")
        
        # Compare responses using embeddings
        if len(responses) < 2:
            return {
                'method': 'cross_validation',
                'score': 1.0,
                'is_hallucination': False,
                'reason': 'Insufficient responses for cross-validation'
            }
        
        # Calculate pairwise similarities
        from relevance_scorer import RelevanceScorer
        scorer = RelevanceScorer()
        
        similarities = []
        response_embeddings = [self.embedding_model.encode(r).tolist() for r in responses]
        
        for i in range(len(response_embeddings)):
            for j in range(i + 1, len(response_embeddings)):
                sim = scorer.cosine_similarity(response_embeddings[i], response_embeddings[j])
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # If responses are very different (< 0.5 similarity), likely inconsistent/hallucinated
        CROSS_VAL_THRESHOLD = 0.5
        is_hallucination = avg_similarity < CROSS_VAL_THRESHOLD
        
        result = {
            'method': 'cross_validation',
            'score': float(avg_similarity),
            'is_hallucination': is_hallucination,
            'threshold': CROSS_VAL_THRESHOLD,
            'num_responses': len(responses),
            'pairwise_similarities': [float(s) for s in similarities],
            'reason': 'Low consistency between responses' if is_hallucination else 'High consistency between responses'
        }
        
        logger.info(f"Cross-validation: avg_similarity={avg_similarity:.3f}, hallucination={is_hallucination}")
        return result
    
    def detect_hallucination(self, response: str, context_chunks: List[Dict], 
                            query: str, methods: List[str] = ['similarity', 'fact_checking']) -> Dict:
        """
        Comprehensive hallucination detection using multiple methods
        
        Args:
            response: Generated LLM response
            context_chunks: List of retrieved context chunks
            query: Original user query
            methods: List of methods to use ['similarity', 'fact_checking', 'cross_validation']
            
        Returns:
            Comprehensive hallucination detection results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'methods': {},
            'overall': {
                'is_hallucination': False,
                'confidence': 0.0,
                'scores': []
            }
        }
        
        # Run selected detection methods
        if 'similarity' in methods:
            results['methods']['similarity'] = self.similarity_scoring(response, context_chunks)
        
        if 'fact_checking' in methods:
            results['methods']['fact_checking'] = self.fact_checking(response, context_chunks, query)
        
        if 'cross_validation' in methods:
            results['methods']['cross_validation'] = self.cross_validation(response, context_chunks, query)
        
        # Aggregate results
        scores = []
        for method_name, method_result in results['methods'].items():
            if 'score' in method_result:
                scores.append(method_result['score'])
                if method_result.get('is_hallucination', False):
                    results['overall']['is_hallucination'] = True
        
        # Calculate overall confidence (average of scores, lower = more hallucination)
        if scores:
            results['overall']['confidence'] = sum(scores) / len(scores)
            results['overall']['scores'] = scores
        
        # Update baseline metrics
        self.update_baseline_metrics(results)
        
        logger.info(f"Hallucination detection complete: hallucination={results['overall']['is_hallucination']}, "
                   f"confidence={results['overall']['confidence']:.3f}")
        
        return results
    
    def update_baseline_metrics(self, detection_results: Dict):
        """Update baseline metrics for tracking over time"""
        self.baseline_metrics['total_queries'] += 1
        
        if detection_results['overall']['is_hallucination']:
            self.baseline_metrics['hallucination_detected_count'] += 1
        
        # Update average scores
        methods = detection_results.get('methods', {})
        
        if 'similarity' in methods:
            sim_score = methods['similarity'].get('score', 0.0)
            self.baseline_metrics['similarity_scores'].append(sim_score)
            if self.baseline_metrics['similarity_scores']:
                self.baseline_metrics['avg_similarity_score'] = (
                    sum(self.baseline_metrics['similarity_scores']) / 
                    len(self.baseline_metrics['similarity_scores'])
                )
        
        if 'fact_checking' in methods:
            fc_score = methods['fact_checking'].get('score', 0.0)
            self.baseline_metrics['fact_check_scores'].append(fc_score)
            if self.baseline_metrics['fact_check_scores']:
                self.baseline_metrics['avg_fact_check_score'] = (
                    sum(self.baseline_metrics['fact_check_scores']) / 
                    len(self.baseline_metrics['fact_check_scores'])
                )
    
    def get_baseline_metrics(self) -> Dict:
        """Get current baseline metrics"""
        metrics = self.baseline_metrics.copy()
        if metrics['total_queries'] > 0:
            metrics['hallucination_rate'] = (
                metrics['hallucination_detected_count'] / metrics['total_queries']
            )
        else:
            metrics['hallucination_rate'] = 0.0
        return metrics
    
    def log_to_langfuse(self, trace_id: str, query: str, response: str, 
                       context_chunks: List[Dict], detection_results: Dict):
        """
        Log hallucination detection results to Langfuse
        
        Args:
            trace_id: Langfuse trace ID for linking
            query: User query
            response: Generated response
            context_chunks: Retrieved context chunks
            detection_results: Detection results
        """
        if not self.langfuse_enabled or not self.langfuse:
            return
        
        try:
            # Create a trace or use existing trace ID
            trace = self.langfuse.trace(
                id=trace_id,
                name="rag_query_with_hallucination_detection",
                metadata={
                    'query': query,
                    'num_context_chunks': len(context_chunks),
                    'hallucination_detected': detection_results['overall']['is_hallucination'],
                    'confidence': detection_results['overall']['confidence']
                }
            )
            
            # Create generation event for the LLM response
            generation = trace.generation(
                name="rag_response_generation",
                model=GEMINI_MODEL,
                input={"query": query, "context": [c['text'] for c in context_chunks]},
                output={"response": response},
                metadata={
                    'num_context_chunks': len(context_chunks)
                }
            )
            
            # Create scores for each detection method
            for method_name, method_result in detection_results['methods'].items():
                generation.score(
                    name=f"hallucination_{method_name}",
                    value=method_result.get('score', 0.0),
                    comment=method_result.get('reason', ''),
                    metadata={
                        'is_hallucination': method_result.get('is_hallucination', False),
                        **{k: v for k, v in method_result.items() if k not in ['score', 'reason', 'is_hallucination']}
                    }
                )
            
            # Overall hallucination score
            generation.score(
                name="overall_hallucination",
                value=detection_results['overall']['confidence'],
                comment=f"Overall hallucination confidence. Hallucination: {detection_results['overall']['is_hallucination']}",
                metadata={
                    'is_hallucination': detection_results['overall']['is_hallucination'],
                    'methods_used': list(detection_results['methods'].keys())
                }
            )
            
            # Flush to ensure data is sent
            self.langfuse.flush()
            
            logger.info(f"Logged hallucination detection to Langfuse (trace_id: {trace_id})")
            
        except Exception as e:
            logger.error(f"Failed to log to Langfuse: {e}")


# Test implementation
if __name__ == "__main__":
    print("Testing Hallucination Monitor...")
    
    # Create a simple test
    monitor = HallucinationMonitor()
    
    test_context = [
        {'text': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.'},
        {'text': 'Deep learning uses neural networks with multiple layers to process complex patterns.'}
    ]
    
    test_response_good = "Machine learning is a subset of AI that allows computers to learn from data."
    test_response_bad = "Quantum computing uses qubits to process information through superposition."
    
    print("\n=== Testing Good Response ===")
    result_good = monitor.detect_hallucination(
        test_response_good, 
        test_context, 
        "What is machine learning?",
        methods=['similarity']
    )
    print(f"Hallucination detected: {result_good['overall']['is_hallucination']}")
    print(f"Confidence: {result_good['overall']['confidence']:.3f}")
    
    print("\n=== Testing Bad Response ===")
    result_bad = monitor.detect_hallucination(
        test_response_bad,
        test_context,
        "What is machine learning?",
        methods=['similarity']
    )
    print(f"Hallucination detected: {result_bad['overall']['is_hallucination']}")
    print(f"Confidence: {result_bad['overall']['confidence']:.3f}")
    
    print("\n=== Baseline Metrics ===")
    metrics = monitor.get_baseline_metrics()
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
