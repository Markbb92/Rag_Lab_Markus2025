"""
Assignment 4: Complete Streamlit RAG Interface
Students: Complete the TODO sections to build the full application
"""

import streamlit as st
import google.generativeai as genai
from document_processor import DocumentProcessor
from relevance_scorer import RelevanceScorer
from guardrails import Guardrails
from hallucination_monitor import HallucinationMonitor
from config import *
import io
import re
import uuid

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
    
    # Initialize hallucination monitor
    hallucination_monitor = HallucinationMonitor(gemini_model=model)
    
    return doc_processor, scorer, guardrails, model, hallucination_monitor


def extract_text_from_file(uploaded_file) -> str:
    """
    Extract text from various file formats
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Extracted text content
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'txt' or file_extension == 'md':
            # Plain text files
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'pdf':
            # PDF files
            try:
                import PyPDF2
                # Reset file pointer and read into BytesIO for PyPDF2
                uploaded_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Basic PDF text cleanup: fix common issues
                if text:
                    # Remove excessive whitespace but preserve sentence structure
                    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
                    # Fix broken words (common in PDFs)
                    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated line breaks
                
                return text.strip()
            except ImportError:
                raise ImportError("PyPDF2 is required for PDF files. Install it with: pip install PyPDF2")
        
        elif file_extension == 'docx':
            # Word documents
            try:
                from docx import Document
                doc = Document(io.BytesIO(uploaded_file.read()))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            except ImportError:
                raise ImportError("python-docx is required for DOCX files. Install it with: pip install python-docx")
        
        else:
            # Try to decode as UTF-8 text for unknown formats
            uploaded_file.seek(0)  # Reset file pointer
            return uploaded_file.read().decode('utf-8', errors='ignore')
    
    except Exception as e:
        raise Exception(f"Error extracting text from {uploaded_file.name}: {str(e)}")


def generate_rag_response(model, query: str, context_chunks: list) -> str:
    """
    Generate response using Gemini with retrieved context
    
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
    
    if not context_chunks:
        return "No relevant content found to answer your question."
    
    try:
        # Format context chunks into a clear context section
        context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        
        # Create a comprehensive prompt
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
        
        # Generate response using Gemini
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


def main():
    # Initialize components
    doc_processor, scorer, guardrails, gemini_model, hallucination_monitor = initialize_components()
    
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
            type=['txt', 'pdf', 'docx', 'md'],
            help="Upload a document to query (supports TXT, PDF, DOCX, MD)"
        )
        
        if uploaded_file:
            # Process uploaded file
            try:
                # Extract text from file (handles multiple formats)
                text = extract_text_from_file(uploaded_file)
                
                if not text or not text.strip():
                    st.warning("⚠️ The file appears to be empty or no text could be extracted.")
                else:
                    # Show extracted text preview for debugging
                    text_preview = text[:200] + "..." if len(text) > 200 else text
                    with st.expander("📝 Extracted Text Preview", expanded=False):
                        st.text(text_preview)
                        st.caption(f"Total characters: {len(text)}")
                    
                    # Ingest document into the system
                    num_chunks = doc_processor.ingest_document(text, uploaded_file.name)
                    
                    # Show success message
                    st.success(f"✅ Document uploaded successfully! Created {num_chunks} chunks.")
                    
                    # Show updated stats
                    stats = doc_processor.get_collection_stats()
                    st.metric("Total Chunks", stats['total_chunks'])
                
            except ImportError as e:
                st.error(f"❌ Missing dependency: {str(e)}")
                st.info("💡 Install required packages: `pip install PyPDF2 python-docx`")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
        
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
            st.write(f"**Hallucination Similarity Threshold:** {HALLUCINATION_SIMILARITY_THRESHOLD}")
            st.write(f"**Detection Methods:** {', '.join(HALLUCINATION_DETECTION_METHODS)}")
    
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
        # Normalize query (strip whitespace) to ensure consistency
        query = query.strip()
        
        # Step 1: Validate query with guardrails
        st.subheader("🛡️ Guardrails Check")
        with st.spinner("Checking query safety..."):
            validation = guardrails.validate_query(query)
            
            # Display validation results
            if validation['safe']:
                st.success("✅ Query is safe to process")
            else:
                st.error("⚠️ Query flagged by guardrails")
                for warning in validation['warnings']:
                    st.warning(f"• {warning}")
                
                if validation['pii_found']:
                    st.error("🚨 PII detected in query:")
                    for pii_type, instances in validation['pii_found'].items():
                        st.write(f"**{pii_type.title()}:** {', '.join(instances)}")
                
                if validation['toxicity']['is_toxic']:
                    st.error(f"🚨 Toxic content detected (score: {validation['toxicity']['score']:.3f})")
                
                # Stop processing if query is unsafe
                st.stop()
        
        # Step 2: Search for relevant chunks
        st.subheader("🔍 Document Search")
        with st.spinner("Searching relevant content..."):
            # Check if there are any documents in the collection first
            stats = doc_processor.get_collection_stats()
            if stats['total_chunks'] == 0:
                st.error("❌ No documents in the collection. Please upload a document first.")
                st.stop()
            
            # Get raw results before filtering to debug
            query_embedding = scorer.embedding_model.encode(query).tolist()
            raw_results = scorer.collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K_RESULTS
            )
            
            # Calculate scores for all results (before threshold filtering)
            all_scored_results = scorer.calculate_relevance_scores(query, raw_results)
            
            # Filter by threshold
            results = scorer.filter_by_threshold(all_scored_results)
            
            # Display number of results found
            if results:
                st.success(f"✅ Found {len(results)} relevant chunks (filtered from {len(all_scored_results)} total results)")
            else:
                st.warning("⚠️ No relevant content found. Try a different question or upload more documents.")
                # Show diagnostic info with raw scores
                with st.expander("🔍 Diagnostic Information", expanded=False):
                    st.write(f"**Total chunks in collection:** {stats['total_chunks']}")
                    st.write(f"**Similarity threshold:** {SIMILARITY_THRESHOLD}")
                    st.write(f"**Query:** {query}")
                    st.write(f"**Raw results before filtering:** {len(all_scored_results)}")
                    
                    if all_scored_results:
                        st.write("**Top 3 results (before threshold filtering):**")
                        for i, result in enumerate(all_scored_results[:3], 1):
                            st.write(f"{i}. Combined Score: {result['combined_score']:.4f} | "
                                   f"Cosine: {result['cosine_similarity']:.4f} | "
                                   f"Keywords: {result['keyword_overlap']:.4f}")
                            st.caption(f"   Text preview: {result['text'][:100]}...")
                    
                    st.info("💡 **Tips:** Try rephrasing your question, using different keywords, or lowering the similarity threshold in config.py")
                st.stop()
        
        # Step 3: Display search results
        if show_details:
            st.subheader("📄 Retrieved Chunks")
            # Loop through results and display in expandable sections
            for i, result in enumerate(results):
                with st.expander(f"Chunk {i+1} - Score: {result['combined_score']:.3f}"):
                    st.write(result['text'])
                    st.caption(f"Source: {result['source']} | Cosine: {result['cosine_similarity']:.3f} | Keywords: {result['keyword_overlap']:.3f}")
        
        # Step 4: Generate RAG response
        st.subheader("🤖 AI Response")
        with st.spinner("Generating response..."):
            response = generate_rag_response(gemini_model, query, results)
            
            # Display response
            st.write(response)
        
        # Step 5: Hallucination Detection
        st.subheader("🔍 Hallucination Monitoring")
        with st.spinner("Detecting hallucinations..."):
            # Generate trace ID for Langfuse
            trace_id = str(uuid.uuid4())
            
            # Detect hallucinations using configured methods
            detection_results = hallucination_monitor.detect_hallucination(
                response=response,
                context_chunks=results,
                query=query,
                methods=HALLUCINATION_DETECTION_METHODS
            )
            
            # Display hallucination detection results
            is_hallucination = detection_results['overall']['is_hallucination']
            confidence = detection_results['overall']['confidence']
            
            if is_hallucination:
                st.error(f"⚠️ Potential hallucination detected (confidence: {confidence:.3f})")
            else:
                st.success(f"✅ Response appears grounded in context (confidence: {confidence:.3f})")
            
            # Show detailed method results
            with st.expander("📊 Detailed Hallucination Analysis", expanded=False):
                for method_name, method_result in detection_results['methods'].items():
                    st.write(f"**{method_name.replace('_', ' ').title()}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Score", f"{method_result.get('score', 0.0):.3f}")
                    with col2:
                        status = "⚠️ Hallucination" if method_result.get('is_hallucination', False) else "✅ Grounded"
                        st.write(status)
                    st.caption(method_result.get('reason', 'No reason provided'))
                    
                    # Show additional details
                    if method_result.get('unsupported_claims'):
                        st.write("**Unsupported Claims:**")
                        for claim in method_result['unsupported_claims']:
                            st.write(f"- {claim}")
                    
                    if method_result.get('pairwise_similarities'):
                        st.write(f"**Pairwise Similarities:** {[f'{s:.3f}' for s in method_result['pairwise_similarities']]}")
                    
                    st.markdown("---")
            
            # Show baseline metrics
            baseline_metrics = hallucination_monitor.get_baseline_metrics()
            with st.expander("📈 Baseline Metrics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries", baseline_metrics['total_queries'])
                with col2:
                    st.metric("Hallucination Rate", f"{baseline_metrics.get('hallucination_rate', 0.0):.2%}")
                with col3:
                    st.metric("Avg Similarity Score", f"{baseline_metrics.get('avg_similarity_score', 0.0):.3f}")
                with col4:
                    st.metric("Avg Fact-Check Score", f"{baseline_metrics.get('avg_fact_check_score', 0.0):.3f}")
            
            # Log to Langfuse
            try:
                hallucination_monitor.log_to_langfuse(
                    trace_id=trace_id,
                    query=query,
                    response=response,
                    context_chunks=results,
                    detection_results=detection_results
                )
                st.caption(f"📊 Logged to Langfuse (Trace ID: {trace_id})")
            except Exception as e:
                st.caption(f"⚠️ Langfuse logging failed: {str(e)}")
        
        # Step 6: Validate response with guardrails
        st.subheader("✅ Response Validation")
        with st.spinner("Validating response..."):
            response_validation = guardrails.validate_response(response)
            
            # Display validation status
            if response_validation['safe']:
                st.success("✅ Response is safe to display")
            else:
                st.error("⚠️ Response flagged by guardrails")
                for warning in response_validation['warnings']:
                    st.warning(f"• {warning}")
                
                if response_validation['pii_found']:
                    st.error("🚨 PII detected in response - consider sanitizing")
                    sanitized_response = guardrails.sanitize_text(response)
                    st.write("**Sanitized Response:**")
                    st.write(sanitized_response)
                
                if response_validation['toxicity']['is_toxic']:
                    st.error(f"🚨 Toxic content detected in response (score: {response_validation['toxicity']['score']:.3f})")
    
    elif search_button and not query:
        st.warning("Please enter a question first!")
    
    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use:
    1. **Upload a document** in the sidebar (supports TXT, PDF, DOCX, MD files)
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
