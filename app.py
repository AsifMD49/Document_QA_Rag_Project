#5 a

#streamlit run app.py        run this finally
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import (
    load_pdf,
    split_documents
)

from src.rag_pipeline import RAGPipeline


load_dotenv()

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


pipeline = get_pipeline()


st.title("📚 Intelligent Document Q&A")
st.caption(
    "Ask questions across your documents using "
    "Retrieval-Augmented Generation."
)


with st.sidebar:

    st.header("Configuration")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=4
    )

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )


if uploaded_files:

    all_documents = []

    with st.spinner("Processing documents..."):

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_path = temp_file.name

            documents = load_pdf(temp_path)

            for doc in documents:
                doc.metadata["source"] = (
                    uploaded_file.name
                )

            chunks = split_documents(documents)

            all_documents.extend(chunks)

            os.unlink(temp_path)

        pipeline.build_index(all_documents)

    st.success(
        f"Indexed {len(uploaded_files)} document(s) "
        f"into {len(all_documents)} chunks."
    )


question = st.chat_input(
    "Ask something about your documents..."
)


if question:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF first."
        )

    else:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                results = pipeline.retrieve(
                    question,
                    k=top_k
                )

                answer, sources = (
                    pipeline.generate_answer(
                        question,
                        results
                    )
                )

            st.write(answer)

            if sources:

                with st.expander(
                    "🔎 Retrieved Sources"
                ):

                    for source in sources:

                        st.write(
                            f"**{source['source']}** "
                            f"— Page {source['page']} "
                            f"— Score: "
                            f"{source['score']:.4f}"
                        )