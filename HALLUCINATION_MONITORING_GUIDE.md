# Hallucination Monitoring Integration Guide

## Overview

This document explains how hallucination monitoring has been integrated into your RAG system using Langfuse for observability and tracking.

## What is Hallucination Detection?

**Hallucinations** occur when LLMs generate information that:
- Is not supported by the retrieved context
- Is factually incorrect
- Contains claims that cannot be verified from the source documents

## Integration Architecture

### Components Added

1. **`hallucination_monitor.py`** - Core detection module
2. **`config.py`** - Configuration for Langfuse and detection thresholds
3. **`rag_app.py`** - Integration into the main application workflow
4. **`requirements.txt`** - Added Langfuse dependency

## Detection Methods Implemented

### 1. Similarity Scoring (Primary Method)
**How it works:**
- Generates embeddings for both the response and the retrieved context
- Calculates cosine similarity between response and context embeddings
- If similarity < 0.3 (threshold), marks as potential hallucination

**Advantages:**
- Fast (no additional API calls)
- Semantic understanding of content
- Works well for detecting off-topic responses

**Example:**
```
Query: "What is machine learning?"
Context: "Machine learning is a subset of AI..."
Response: "Machine learning is a subset of AI..." → High similarity ✅
Response: "Quantum computing uses qubits..." → Low similarity ❌
```

### 2. Fact-Checking (Secondary Method)
**How it works:**
- Uses Gemini LLM itself as a fact-checker
- Sends response + context to LLM with fact-checking prompt
- LLM analyzes if all claims in response are supported by context
- Returns JSON with support status, confidence, and unsupported claims

**Advantages:**
- Deep semantic understanding
- Can identify specific unsupported claims
- Handles nuanced fact-checking

**Disadvantages:**
- Requires additional API calls (slower, costs more)
- Depends on LLM quality

**Example Output:**
```json
{
    "supported": false,
    "confidence": 0.2,
    "unsupported_claims": ["Response mentions quantum computing which is not in context"],
    "reason": "Claims about quantum computing cannot be verified in context"
}
```

### 3. Cross-Validation (Optional Method)
**How it works:**
- Generates multiple responses to the same query
- Compares all responses for consistency using embeddings
- Low consistency (< 0.5 similarity) indicates potential hallucination

**Advantages:**
- Detects inconsistency-based hallucinations
- Good for catching random/inaccurate responses

**Disadvantages:**
- Very slow (requires 2+ additional API calls)
- Expensive
- Not always reliable

## How It Integrates with Your System

### Workflow Integration

```
User Query
    ↓
[Existing Steps 1-4]
    ↓
Step 4: Generate RAG Response (Gemini)
    ↓
Step 5: Hallucination Detection (NEW)
    ├─ Similarity Scoring
    ├─ Fact-Checking (if enabled)
    └─ Cross-Validation (optional)
    ↓
Step 6: Guardrails Validation (existing)
    ↓
Display Results
```

### Integration Points in `rag_app.py`

1. **Initialization** (Line ~24):
   ```python
   hallucination_monitor = HallucinationMonitor(gemini_model=model)
   ```

2. **Detection** (After response generation):
   ```python
   detection_results = hallucination_monitor.detect_hallucination(
       response=response,
       context_chunks=results,
       query=query,
       methods=HALLUCINATION_DETECTION_METHODS
   )
   ```

3. **Logging to Langfuse**:
   ```python
   hallucination_monitor.log_to_langfuse(
       trace_id=trace_id,
       query=query,
       response=response,
       context_chunks=results,
       detection_results=detection_results
   )
   ```

## Langfuse Integration

### What is Langfuse?

Langfuse is an open-source observability platform for LLM applications that provides:
- **Traces**: Full request/response cycles
- **Generations**: Individual LLM calls with metadata
- **Scores**: Metrics and evaluations
- **Analytics**: Dashboard with baseline metrics

### Setup Instructions

1. **Get Langfuse API Keys:**
   - Sign up at https://cloud.langfuse.com (free tier available)
   - Create a new project
   - Get your `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from project settings

2. **Set Environment Variables:**
   ```bash
   # Windows PowerShell
   $env:LANGFUSE_PUBLIC_KEY="your-public-key"
   $env:LANGFUSE_SECRET_KEY="your-secret-key"
   
   # Linux/Mac
   export LANGFUSE_PUBLIC_KEY="your-public-key"
   export LANGFUSE_SECRET_KEY="your-secret-key"
   ```

   OR update `config.py` directly:
   ```python
   LANGFUSE_PUBLIC_KEY = "your-public-key"
   LANGFUSE_SECRET_KEY = "your-secret-key"
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Your App:**
   ```bash
   streamlit run rag_app.py
   ```

### What Gets Logged to Langfuse?

For each query, Langfuse receives:
- **Trace**: Complete request/response cycle
  - User query
  - Number of context chunks
  - Hallucination detection results
  
- **Generation**: The LLM response
  - Input (query + context)
  - Output (generated response)
  - Model used
  
- **Scores**: Multiple metrics
  - `hallucination_similarity`: Similarity scoring result (0-1)
  - `hallucination_fact_checking`: Fact-checking result (0-1)
  - `overall_hallucination`: Combined confidence score (0-1)

## Configuration

### `config.py` Settings

```python
# Hallucination Detection Configuration
HALLUCINATION_SIMILARITY_THRESHOLD = 0.3  # Below this = likely hallucination
HALLUCINATION_FACT_CHECK_ENABLED = True   # Enable fact-checking
HALLUCINATION_CROSS_VAL_ENABLED = False   # Enable cross-validation (slower)
HALLUCINATION_DETECTION_METHODS = ['similarity', 'fact_checking']  # Methods to use
```

**Tuning Thresholds:**
- **Lower threshold (e.g., 0.2)**: More lenient, fewer false positives, but may miss some hallucinations
- **Higher threshold (e.g., 0.4)**: More strict, catches more potential hallucinations, but may have false positives

### Recommended Settings

**For Production:**
- Use `['similarity']` only (fast, cost-effective)
- Threshold: `0.3`

**For Development/Testing:**
- Use `['similarity', 'fact_checking']` (more thorough)
- Threshold: `0.3`
- Enable fact-checking for detailed analysis

**For Research:**
- Use all methods including cross-validation
- Lower threshold: `0.25`
- Full logging enabled

## Baseline Metrics

The system tracks the following metrics over time:

### Metrics Collected

1. **Total Queries**: Number of queries processed
2. **Hallucination Detected Count**: Number of queries flagged as hallucinations
3. **Hallucination Rate**: Percentage of queries with hallucinations
4. **Average Similarity Score**: Mean similarity score across all queries
5. **Average Fact-Check Score**: Mean fact-check score across all queries

### Accessing Metrics

In the Streamlit UI:
- Click "📈 Baseline Metrics" expander to see current stats

Programmatically:
```python
baseline_metrics = hallucination_monitor.get_baseline_metrics()
print(f"Hallucination Rate: {baseline_metrics['hallucination_rate']:.2%}")
```

In Langfuse Dashboard:
- Navigate to your project
- View scores and analytics in the dashboard
- Filter by time period, model, etc.

## Logging

### File Logging

Logs are written to `hallucination_monitor.log` with:
- Timestamp
- Detection method used
- Scores and results
- Errors and warnings

### Log Levels

- **INFO**: Normal operations, detection results
- **WARNING**: Langfuse unavailable, methods skipped
- **ERROR**: Detection failures, API errors

## Usage Examples

### Example 1: Normal Query (No Hallucination)

```
Query: "What is machine learning?"
Context: ["Machine learning is a subset of AI..."]
Response: "Machine learning is a subset of artificial intelligence..."

Results:
- Similarity Score: 0.85 ✅ (High similarity)
- Fact-Check Score: 0.95 ✅ (Fully supported)
- Overall: No hallucination detected
```

### Example 2: Potential Hallucination

```
Query: "What is machine learning?"
Context: ["Machine learning is a subset of AI..."]
Response: "Quantum computing revolutionizes machine learning through qubits..."

Results:
- Similarity Score: 0.15 ❌ (Low similarity)
- Fact-Check Score: 0.20 ❌ (Unsupported: quantum computing not in context)
- Overall: Hallucination detected
- Unsupported Claims: ["Response mentions quantum computing which is not in context"]
```

## Troubleshooting

### Langfuse Not Logging

1. **Check API Keys:**
   ```python
   import os
   print(os.getenv("LANGFUSE_PUBLIC_KEY"))
   print(os.getenv("LANGFUSE_SECRET_KEY"))
   ```

2. **Check Langfuse Status:**
   - App will show warning if Langfuse unavailable
   - Detection still works without Langfuse (just no logging)

3. **Check Network:**
   - Ensure internet connection
   - Check firewall settings

### Fact-Checking Failing

1. **Check Gemini API Key:**
   - Required for fact-checking
   - Without it, similarity scoring still works

2. **Check API Quotas:**
   - Fact-checking uses additional API calls
   - May hit rate limits

### High False Positive Rate

1. **Lower Threshold:**
   ```python
   HALLUCINATION_SIMILARITY_THRESHOLD = 0.2  # More lenient
   ```

2. **Use Multiple Methods:**
   - Requires both methods to flag for more accuracy
   - Modify `detect_hallucination()` aggregation logic

## Performance Considerations

### Speed Impact

- **Similarity Scoring**: ~50-100ms (minimal impact)
- **Fact-Checking**: ~1-2 seconds (one additional API call)
- **Cross-Validation**: ~3-6 seconds (multiple API calls)

### Cost Impact

- **Similarity Scoring**: Free (local computation)
- **Fact-Checking**: ~$0.001-0.002 per query (Gemini API)
- **Cross-Validation**: ~$0.003-0.006 per query (multiple Gemini API calls)

### Recommendations

- Use similarity scoring for all queries (fast, free)
- Use fact-checking selectively (when quality is critical)
- Use cross-validation only for testing/evaluation

## Future Enhancements

Potential improvements:
1. **Self-Consistency Scoring**: Compare response to query directly
2. **Citation Tracking**: Verify all facts have citations
3. **Confidence Calibration**: Learn from user feedback
4. **Adaptive Thresholds**: Adjust based on document type
5. **Batch Processing**: Process multiple queries efficiently

## Documentation Requirements

For your assignment, document:

1. **Approach**: Which methods you implemented and why
2. **Configuration**: Thresholds and settings used
3. **Baseline Metrics**: Initial results (first 10-20 queries)
4. **Findings**: What you learned about hallucinations in your system
5. **Langfuse Screenshots**: Dashboard showing traces and scores
6. **Challenges**: Issues encountered and how resolved

## Summary

You now have a complete hallucination monitoring system that:
- ✅ Detects hallucinations using multiple methods
- ✅ Integrates seamlessly with your existing RAG pipeline
- ✅ Logs everything to Langfuse for observability
- ✅ Tracks baseline metrics over time
- ✅ Provides detailed analysis in the UI

The system is production-ready and can be tuned based on your specific needs and document types.
