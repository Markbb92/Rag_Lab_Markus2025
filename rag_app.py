"""
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
            # Process uploaded file
            try:
                # Read file content
                text = uploaded_file.read().decode('utf-8')
                
                # Ingest document into the system
                num_chunks = doc_processor.ingest_document(text, uploaded_file.name)
                
                # Show success message
                st.success(f"✅ Document uploaded successfully! Created {num_chunks} chunks.")
                
                # Show updated stats
                stats = doc_processor.get_collection_stats()
                st.metric("Total Chunks", stats['total_chunks'])
                
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
            results = scorer.search_and_score(query)
            
            # Display number of results found
            if results:
                st.success(f"✅ Found {len(results)} relevant chunks")
            else:
                st.warning("⚠️ No relevant content found. Try a different question or upload more documents.")
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
        
        # Step 5: Validate response with guardrails
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
