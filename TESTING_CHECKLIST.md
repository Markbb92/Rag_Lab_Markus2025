# Testing Checklist for Hallucination Monitoring

## Pre-Testing Setup

### ✅ 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit
- chromadb
- sentence-transformers
- google-generativeai
- transformers
- langfuse (NEW)
- python-dotenv (NEW)

### ✅ 2. Configure API Keys

**Option A: Using .env file (Recommended)**
1. Open `.env` file in project root
2. Add your keys:
   ```
   GEMINI_API_KEY=your-gemini-key
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   ```

**Option B: Environment Variables**
```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
```

### ✅ 3. Verify Configuration
- [ ] `config.py` has `load_dotenv()` call
- [ ] `.env` file exists (or environment variables set)
- [ ] Langfuse keys are set (optional - system works without them)

## Testing Steps

### Test 1: Basic Functionality (Without Langfuse)
**Goal:** Verify hallucination detection works even without Langfuse

1. **Start the app:**
   ```bash
   streamlit run rag_app.py
   ```

2. **Upload a document:**
   - Use a test document (e.g., `sample_documents/ai_history.txt`)
   - Verify document uploads successfully

3. **Ask a question:**
   - Query: "What is this document about?"
   - Click "🔍 Search"

4. **Check results:**
   - [ ] Response is generated
   - [ ] "🔍 Hallucination Monitoring" section appears
   - [ ] Similarity score is displayed
   - [ ] Baseline metrics show up
   - [ ] Warning about Langfuse (if keys not set) - this is OK

**Expected:** System works, shows similarity scoring results

---

### Test 2: With Langfuse (Full Integration)
**Goal:** Verify Langfuse logging works

1. **Set Langfuse keys** in `.env` or environment variables

2. **Restart the app**

3. **Run a query:**
   - Upload document
   - Ask a question
   - Check results

4. **Verify Langfuse:**
   - [ ] No Langfuse error messages
   - [ ] Trace ID is displayed in UI
   - [ ] Check Langfuse dashboard at https://cloud.langfuse.com
   - [ ] Verify trace appears in dashboard

**Expected:** Full logging to Langfuse, trace visible in dashboard

---

### Test 3: Hallucination Detection - Good Response
**Goal:** Verify system correctly identifies grounded responses

1. **Upload a document** with clear content (e.g., about AI/ML)

2. **Ask a question** that can be answered from the document:
   - Example: "What is machine learning?" (if doc contains ML info)

3. **Check detection results:**
   - [ ] Similarity score > 0.3 (should be high)
   - [ ] Status shows "✅ Response appears grounded"
   - [ ] Fact-checking score (if enabled) should be high
   - [ ] No hallucination detected

**Expected:** High scores, no hallucination flag

---

### Test 4: Hallucination Detection - Bad Response
**Goal:** Verify system detects hallucinations

**Note:** This is harder to test naturally. You can:
- Ask about something NOT in the document
- Or manually test the detection logic

1. **Upload a document** about a specific topic (e.g., "History of AI")

2. **Ask about something unrelated:**
   - Example: "What is quantum computing?" (if doc doesn't mention it)

3. **Check detection results:**
   - [ ] Similarity score < 0.3 (should be low)
   - [ ] Status shows "⚠️ Potential hallucination detected"
   - [ ] Detailed analysis shows low scores

**Expected:** Low scores, hallucination flag

---

### Test 5: Baseline Metrics
**Goal:** Verify metrics tracking

1. **Run multiple queries** (3-5 different questions)

2. **Check baseline metrics:**
   - [ ] Total Queries increases
   - [ ] Hallucination Rate updates
   - [ ] Average scores are calculated
   - [ ] Metrics persist across queries

**Expected:** Metrics accumulate correctly

---

### Test 6: Error Handling
**Goal:** Verify graceful error handling

1. **Test without Gemini API key:**
   - [ ] App starts (with warning)
   - [ ] Similarity scoring still works
   - [ ] Fact-checking shows "model not available" message

2. **Test without Langfuse keys:**
   - [ ] App works normally
   - [ ] Warning message appears
   - [ ] Detection still functions

3. **Test with empty document:**
   - [ ] Appropriate error messages
   - [ ] No crashes

**Expected:** Graceful degradation, informative error messages

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'langfuse'"
**Solution:** 
```bash
pip install langfuse>=2.0.0
```

### Issue: "ModuleNotFoundError: No module named 'dotenv'"
**Solution:**
```bash
pip install python-dotenv>=1.0.0
```

### Issue: Langfuse initialization fails
**Solution:**
- Check API keys are correct
- Check internet connection
- System will work without Langfuse (just no logging)

### Issue: Fact-checking always fails
**Solution:**
- Verify Gemini API key is set
- Check API quota/limits
- Similarity scoring works without fact-checking

### Issue: No hallucination detection results
**Solution:**
- Check that `HALLUCINATION_DETECTION_METHODS` in config.py includes 'similarity'
- Verify documents are uploaded
- Check console for error messages

---

## Success Criteria

✅ **Minimum Viable Test:**
- App starts without errors
- Document upload works
- Query processing works
- Hallucination monitoring section appears
- Similarity scoring shows results

✅ **Full Integration Test:**
- All above, plus:
- Langfuse logging works
- Baseline metrics track correctly
- Detailed analysis expandable sections work
- No crashes or errors

---

## Quick Test Command

Run this to test the hallucination monitor directly:
```bash
python hallucination_monitor.py
```

This will run the built-in test and show if the module works correctly.

---

## Next Steps After Testing

1. **Document your findings:**
   - Baseline metrics from first 10-20 queries
   - Hallucination rate observed
   - Any issues encountered

2. **Take screenshots:**
   - Langfuse dashboard showing traces
   - UI showing detection results
   - Baseline metrics

3. **Tune thresholds** if needed:
   - Adjust `HALLUCINATION_SIMILARITY_THRESHOLD` in config.py
   - Test different detection methods
