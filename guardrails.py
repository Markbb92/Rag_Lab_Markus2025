"""
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
        Detect personally identifiable information in text
        
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
        
        # Loop through PII patterns from config
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found_pii[pii_type] = matches
        
        return found_pii
    
    def check_toxicity(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Check text for toxic content using ML model
        
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
        
        try:
            # Use toxicity detector (limit to 512 chars for model efficiency)
            results = self.toxicity_detector(text[:512])
            
            # Find the 'toxic' label and extract score
            max_score = 0.0
            is_toxic = False
            details = {}
            
            for result in results:
                if result['label'] == 'toxic':
                    max_score = result['score']
                    is_toxic = max_score > TOXICITY_THRESHOLD
                    details = {
                        'toxic_score': max_score,
                        'threshold': TOXICITY_THRESHOLD,
                        'all_scores': results
                    }
                    break
            
            return is_toxic, max_score, details
            
        except Exception as e:
            # Handle any errors in toxicity detection
            return False, 0.0, {"error": str(e), "note": "Toxicity detection failed"}
    
    def validate_query(self, query: str) -> Dict:
        """
        Complete validation check for user queries
        
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
        
        # 1. Check for PII
        pii = self.detect_pii(query)
        if pii:
            result['safe'] = False
            result['warnings'].append("PII detected in query")
            result['pii_found'] = pii
        
        # 2. Check toxicity
        is_toxic, score, details = self.check_toxicity(query)
        if is_toxic:
            result['safe'] = False
            result['warnings'].append("Toxic content detected")
            result['toxicity'] = {'is_toxic': True, 'score': score, 'details': details}
        
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
        Remove or mask PII from text
        
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
        
        # Find PII in the text
        pii = self.detect_pii(text)
        
        # Replace each PII instance with masked version
        for pii_type, instances in pii.items():
            for instance in instances:
                sanitized = sanitized.replace(instance, f"[REDACTED_{pii_type.upper()}]")
        
        return sanitized
    
    def print_validation_result(self, result: Dict, text_preview: str = ""):
        """Pretty print validation results"""
        print("\n" + "="*80)
        print("GUARDRAILS VALIDATION RESULT")
        print("="*80)
        
        if text_preview:
            print(f"\nText preview: {text_preview[:100]}...")
        
        status = "✓ SAFE" if result['safe'] else "⚠ FLAGGED"
        print(f"\nStatus: {status}")
        
        if result['warnings']:
            print(f"\nWarnings:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        if result['pii_found']:
            print(f"\nPII Detected:")
            for pii_type, instances in result['pii_found'].items():
                print(f"  - {pii_type}: {len(instances)} instance(s)")
                for instance in instances[:3]:  # Show first 3
                    print(f"    * {instance}")
        
        if result['toxicity']['is_toxic']:
            print(f"\nToxicity:")
            print(f"  - Score: {result['toxicity']['score']:.3f}")
            print(f"  - Threshold: {TOXICITY_THRESHOLD}")
        
        print("="*80 + "\n")


# Test your implementation
if __name__ == "__main__":
    print("Initializing Guardrails (may download model on first run)...")
    guardrails = Guardrails()
    
    # Test PII detection
    print("\n" + "="*80)
    print("TEST 1: PII Detection")
    print("="*80)
    
    test_texts = [
        "My email is john.doe@example.com and phone is 555-123-4567",
        "Contact me at alice@company.org or call 123-456-7890",
        "SSN: 123-45-6789, Card: 4532-1234-5678-9010",
        "This text has no PII at all"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        pii = guardrails.detect_pii(text)
        if pii:
            print(f"Found PII: {pii}")
            sanitized = guardrails.sanitize_text(text)
            print(f"Sanitized: {sanitized}")
        else:
            print("No PII detected")
    
    # Test toxicity detection
    print("\n" + "="*80)
    print("TEST 2: Toxicity Detection")
    print("="*80)
    
    if guardrails.toxicity_enabled:
        toxic_texts = [
            "This is a friendly and helpful message",
            "You are stupid and worthless",
            "I disagree with your opinion, but respect your view"
        ]
        
        for text in toxic_texts:
            print(f"\nText: {text}")
            is_toxic, score, details = guardrails.check_toxicity(text)
            print(f"Toxic: {is_toxic}, Score: {score:.3f}")
    else:
        print("Toxicity detection disabled")
    
    # Test full query validation
    print("\n" + "="*80)
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
