#1  "I used recursive character splitting because fixed character splitting 
# can cut semantic units arbitrarily.
# The recursive splitter tries larger separators first and gradually falls back to smaller ones."
from PIL import GifImagePlugin
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks