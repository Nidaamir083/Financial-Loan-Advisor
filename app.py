import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import os
from PIL import Image
import tempfile

# Set page config
st.set_page_config(
    page_title="Loan Pilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Knowledge Base
knowledge_chunks = [
    {"text": "A good credit score (650+) is essential for loan approval.", "source": "credit_requirements"},
    {"text": "Down payments usually range from 10% to 20% for home loans.", "source": "loan_terms"},
    {"text": "Loan eligibility is based on monthly income, liabilities, and credit history.", "source": "eligibility"},
    {"text": "Interest rates vary by bank, typically between 7% to 11% annually.", "source": "interest_rates"},
    {"text": "For home loans, you typically need salary slips, bank statements, tax returns, and ID proof.", "source": "documentation"},
    {"text": "For business loans, you need business registration documents, financial statements, and tax returns.", "source": "documentation"},
    {"text": "Loan tenure can be between 5 to 30 years depending on the loan type.", "source": "loan_terms"},
    {"text": "SME loans often require business plans and cash flow projections.", "source": "documentation"},
    {"text": "The debt-to-income ratio should typically be below 40% for loan approval.", "source": "eligibility"},
    {"text": "Prepayment penalties may apply if you pay off your loan early.", "source": "loan_terms"},
    {"text": "Fixed interest rates remain the same throughout the loan term.", "source": "interest_rates"},
    {"text": "Variable interest rates may change based on market conditions.", "source": "interest_rates"}
]

# Initialize models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create FAISS index
    embeddings = embedder.encode([chunk["text"] for chunk in knowledge_chunks])
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return embedder, index

embedder, faiss_index = load_models()

# Document processing functions
def pdf_to_image(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    return images[0] if images else None

def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text_from_image(image, lang='eng+urd'):
    try:
        processed = preprocess_image(image)
        text = pytesseract.image_to_string(processed, lang=lang)
        return text
    except Exception as e:
        return f"OCR Failed: {e}"

# Response generation
def generate_response(user_query, top_k=3):
    query_embedding = embedder.encode([user_query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    relevant_chunks = [knowledge_chunks[idx] for idx in indices[0]]
    
    response = "Here's what I found:\n\n"
    for i, chunk in enumerate(relevant_chunks, 1):
        response += f"{i}. {chunk['text']} (Source: {chunk['source']})\n\n"
    
    if distances[0][0] > 1.0:
        response = "I couldn't find exact information, but here are some potentially related points:\n\n" + response
    
    return response

# UI Components
def sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Loan+Advisor", width=150)
        st.title("Navigation")
        
        app_mode = st.radio(
            "Select Mode",
            ["Chat Advisor", "Document Analysis"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI Loan Advisor helps you:
        - Get information about loan requirements
        - Analyze your loan documents
        - Understand eligibility criteria
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ by [Your Name]")
    
    return app_mode

def chat_advisor():
    st.header("💬 Chat with Loan Advisor")
    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about loan requirements..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        response = generate_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)

def document_analysis():
    st.header("📄 Document Analysis")
    
    uploaded_file = st.file_uploader("Upload your loan document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process document
        with st.spinner("Processing document..."):
            img = pdf_to_image(tmp_path)
            if img:
                st.image(img, caption="First page of your document", width=300)
                ocr_text = extract_text_from_image(img)
                
                with st.expander("View extracted text"):
                    st.text(ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text)
                
                # Clean up
                os.unlink(tmp_path)
                
                # Chat about document
                st.subheader("Ask about your document")
                
                doc_prompt = st.text_input("Ask questions about your uploaded document")
                if doc_prompt and st.button("Ask"):
                    response = generate_response(doc_prompt)
                    st.info(response)
            else:
                st.error("Failed to process the PDF file")

# Main App
def main():
    # Sidebar navigation
    app_mode = sidebar()
    
    # Main content
    if app_mode == "Chat Advisor":
        chat_advisor()
    else:
        document_analysis()

if __name__ == "__main__":
    main()
