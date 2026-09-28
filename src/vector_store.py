#2     Important interview concept

#Your interviewer may ask:

#"Why FAISS?"

#Answer:

#  "FAISS provides efficient vector similarity search.
# For this project I used it as an in-memory/local vector store because,
#  the application is a lightweight document-Q&A system.
# For production scale, I would consider a managed or persistent vector database 
#                                                            depending on the requirements."

#That's much better than:

#"FAISS is a vector database."
#Technically, FAISS is primarily a similarity-search library, not a full database system.


from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 


class VectorStore:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.store = None

    def create(self, documents: List[Document]):
        self.store = FAISS.from_documents(
            documents,
            self.embeddings
        )

    def search(
        self,
        query: str,
        k: int = 4
    ):
        if self.store is None:
            return []

        return self.store.similarity_search_with_score(
            query,
            k=k
        )