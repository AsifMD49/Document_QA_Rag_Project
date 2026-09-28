#5 b

#streamlit run app.py        run this finally
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Document Q&A")
st.caption("Testing application startup")

st.write("Step 1: Streamlit loaded")

from src.document_processor import load_pdf, split_documents

st.write("Step 2: document_processor imported")

from src.rag_pipeline import RAGPipeline

st.write("Step 3: rag_pipeline imported")


@st.cache_resource
def get_pipeline():
    st.write("Creating RAG pipeline...")
    return RAGPipeline()


pipeline = get_pipeline()

st.success("Pipeline created successfully!")

with st.sidebar:
    st.header("Configuration")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
