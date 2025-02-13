import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A ChatBot"
groq_api = os.getenv('GROQ_API_KEY')

# Load model
llm = ChatGroq(groq_api_key=groq_api, model_name="Llama-3.3-70b-Specdec")

# Define the chatbot prompt
prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on the given context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    <context>
    
    Question: {input}
""")

# App title with styling
st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">📖 RAG Document Q&A ChatBot</h1>
    <p style="text-align:center; font-size:16px; color:#5D6D7E;">
        Upload a PDF Document, generate embeddings, and ask questions based on its content.
    </p>
    """,
    unsafe_allow_html=True
)

# Function to process uploaded document and create vector embeddings
def create_vector_embeddings(pdf_file):
    if "vectors" not in st.session_state:
        if pdf_file is not None:
            with st.spinner("🔄 Processing document... Please wait."):

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(pdf_file.read())
                    temp_pdf_path = temp_pdf.name
                
                # Load document using PyMuPDFLoader
                pdf_reader = PyMuPDFLoader(temp_pdf_path)
                docs = pdf_reader.load()

                # Create embeddings
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_docs = text_splitter.split_documents(docs)

                # Store vectors in FAISS
                st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)
                
                # Delete the temporary file
                os.remove(temp_pdf_path)
                
                st.success("✅ Vector Database is ready!")

# Sidebar for document upload & vector generation inside a slider
with st.expander("📂 **Upload Document & Generate Vectors**", expanded=False):
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if st.button("Generate Embeddings", use_container_width=True):
        create_vector_embeddings(uploaded_file)

# Separator for UI clarity
st.markdown("<hr style='border: 1px solid #444; margin-bottom: 15px;'>", unsafe_allow_html=True)

# User query input
st.markdown("### 🔍 Ask a Question from the Document")
user_prompt = st.text_input("Enter your query here:")

# Get answer from the chatbot
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("🤖 Generating response..."):
        response = retrieval_chain.invoke({"input": user_prompt})

    # Display answer with proper spacing
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 8px;
            color: #FFFFFF;
            font-size: 16px;
            margin-bottom: 20px;">
            🤖 <b>Response:</b> {response['answer']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add separator to avoid collisions
    st.markdown("<hr style='border: 1px solid #444; margin-bottom: 15px;'>", unsafe_allow_html=True)

    # Document Similarity Search Section
    with st.expander("📚 **Document Similarity Search**", expanded=False):
        for i, doc in enumerate(response['context']):
            st.markdown(
                f"""
                <div style="
                    background-color: #262626;
                    padding: 10px;
                    border-radius: 6px;
                    color: #DDDDDD;
                    font-size: 14px;
                    margin-bottom: 10px;">
                    {doc.page_content}
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer with better message
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#5D6D7E;">
    🚀 Built for intelligent document exploration using <b>LangChain</b>, <b>Streamlit</b>, and <b>Groq AI</b>.
    </p>
    """,
    unsafe_allow_html=True
)

