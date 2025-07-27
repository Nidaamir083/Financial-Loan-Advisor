import sys
import logging

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # All your existing imports here
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except ImportError as e:
    logger.error(f"Import failed: {e}")
    st.error(f"Required package failed to load: {e}")
    
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
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from io import BytesIO

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
    {"text": "Variable interest rates may change based on market conditions.", "source": "interest_rates"},
    {"text": "SME loans typically require a minimum business revenue of $100,000/year for eligibility.", "source": "sme_eligibility"},
    {"text": "Startups may need to provide a 2-year projected financial statement instead of historical records.", "source": "sme_documentation"},
    {"text": "Collateral for business loans can include commercial property, equipment, or inventory.", "source": "sme_collateral"},
    {"text": "Banks may require a personal guarantee from business owners for SME loans.", "source": "sme_requirements"},
    {"text": "Industry-specific risk assessments (e.g., hospitality vs. manufacturing) affect loan terms.", "source": "sme_risk"},
    {"text": "Joint home loan applications can increase eligibility based on combined incomes.", "source": "home_loan_terms"},
    {"text": "Home loans for properties under construction have staggered disbursement schedules.", "source": "home_loan_disbursement"},
    {"text": "Government-backed loans (e.g., FHA, VA) may have lower down payment requirements.", "source": "government_loans"},
    {"text": "Home loan EMI should not exceed 40% of your monthly income.", "source": "home_loan_eligibility"},
    {"text": "Property insurance is mandatory for home loans in flood-prone areas.", "source": "home_loan_insurance},
    {"text": "Late payments on existing loans can reduce your credit score by 50-100 points.", "source": "credit_impact"},
    {"text": "A debt-to-income ratio above 50% usually leads to loan rejection.", "source", "risk_assessment"},
    {"text": "Banks may offer lower interest rates to customers with long-term banking relationships.", "source": "loyalty_benefits"},
    {"text": "Loan rejections within the last 6 months negatively impact future applications.", "source": "credit_history"},
    {"text": "Credit score ranges: Poor (<580), Fair (580-669), Good (670-739), Excellent (740+).", "source", "credit_score_ranges"},
    {"text": "Document processing times range from 3-7 business days for most loans.", "source": "processing_time"},
    {"text": "Digital document submission can accelerate approval by 30%.", "source": "digital_process"},
    {"text": "Notarized affidavits may be required for address verification.", "source": "documentation"},
    {"text": "Loan sanction letters are valid for 3-6 months depending on the bank.", "source": "sanction_terms"},
    {"text": "Physical signatures are still required for mortgage agreements in most countries.", "source": "legal_requirements"},
    {"text": "Medical emergency loans may require hospital admission proof but have faster approval.", "source": "emergency_loans"},
    {"text": "Green energy loans often come with 0.5-1% interest rate subsidies.", "source": "green_loans"},
    {"text": "Loans for women entrepreneurs may have 10-20% lower interest rates.", "source": "gender_based_loans"},
    {"text": "Agricultural loans may offer monsoon-season repayment holidays.", "source": "agricultural_loans"}

}
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

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyPDF2 (faster but less accurate)"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returns
        return text
    except Exception as e:
        st.error(f"PyPDF2 extraction failed: {e}")
        return None

def extract_text_from_pdf_miner(pdf_path):
    """Extract text from PDF using pdfminer (slower but more accurate)"""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        st.error(f"PDFMiner extraction failed: {e}")
        return None

def analyze_pdf_content(text):
    """Analyze extracted text for key loan-related information"""
    analysis = {
        "keywords_found": [],
        "potential_missing": []
    }
    
    # List of important keywords to look for
    important_keywords = [
        "income", "salary", "employment", "credit score",
        "debt", "loan amount", "property", "collateral",
        "tax returns", "bank statements", "identification"
    ]
    
    for keyword in important_keywords:
        if keyword.lower() in text.lower():
            analysis["keywords_found"].append(keyword)
    
    # Check for common missing elements
    common_missing = []
    if "bank statements" not in analysis["keywords_found"]:
        common_missing.append("bank statements")
    if "tax returns" not in analysis["keywords_found"]:
        common_missing.append("tax returns")
    
    analysis["potential_missing"] = common_missing
    
    return analysis

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
        st.markdown("Made by [Team]")
    
    return app_mode

def chat_advisor():
    st.header("💬 Chat with Loan Pilot")
    
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

def pdf_analysis_section():
    st.header("📄 PDF Document Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload your loan application PDF", 
        type=["pdf"],
        help="Upload your completed loan application form or supporting documents"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Document Preview")
            
            # Show PDF metadata
            try:
                pdf_reader = PdfReader(uploaded_file)
                st.write(f"**Pages:** {len(pdf_reader.pages)}")
                if pdf_reader.metadata:
                    st.write("**Metadata:**")
                    st.json(pdf_reader.metadata)
            except:
                pass
            
            # PDF preview (first page)
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1)
                st.image(images[0], caption="First page preview", use_column_width=True)
            except Exception as e:
                st.warning(f"Couldn't generate preview: {e}")
        
        with col2:
            st.subheader("Document Analysis")
            
            # Create tabs for different analysis methods
            tab1, tab2 = st.tabs(["Quick Analysis", "Detailed Analysis"])
            
            with tab1:
                with st.spinner("Extracting text (fast method)..."):
                    text = extract_text_from_pdf(uploaded_file)
                
                if text:
                    st.success("Text extracted successfully!")
                    with st.expander("View extracted text"):
                        st.text(text[:2000] + "..." if len(text) > 2000 else text)
                    
                    # Basic analysis
                    analysis = analyze_pdf_content(text)
                    st.subheader("Key Information Found")
                    if analysis["keywords_found"]:
                        st.write(", ".join(analysis["keywords_found"]))
                    else:
                        st.warning("No key loan terms found in document")
                    
                    st.subheader("Potential Missing Information")
                    if analysis["potential_missing"]:
                        st.warning("The following items may be missing: " + ", ".join(analysis["potential_missing"]))
                    else:
                        st.success("All key loan application elements detected")
            
            with tab2:
                # Save to temp file for pdfminer
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                with st.spinner("Extracting text (detailed method)..."):
                    detailed_text = extract_text_from_pdf_miner(tmp_path)
                    os.unlink(tmp_path)  # Clean up temp file
                
                if detailed_text:
                    st.success("Detailed text extraction complete!")
                    with st.expander("View detailed extracted text"):
                        st.text(detailed_text[:2000] + "..." if len(detailed_text) > 2000 else detailed_text)
                    
                    # More detailed analysis could go here
                    st.info("Detailed analysis would compare document content against loan requirements")

        # Add a section for asking questions about the document
        st.divider()
        st.subheader("Ask About Your Document")
        
        doc_question = st.text_input(
            "Ask specific questions about your uploaded document",
            placeholder="What information is missing from my application?"
        )
        
        if doc_question and st.button("Get Answer"):
            # Combine document text with question for context
            context = f"Document content:\n{text[:3000]}\n\nQuestion: {doc_question}"
            response = generate_response(context)
            
            st.markdown("### Pilot Response")
            st.markdown(response)

# Main App
def main():
    # Sidebar navigation
    app_mode = sidebar()
    
    # Main content
    if app_mode == "Loan Pilot":
        chat_advisor()
    elif app_mode == "Document Analysis":
        pdf_analysis_section()

if __name__ == "__main__":
    main()

   



      



   
        
       
        
      
  
