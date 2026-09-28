#4
from typing import List, Tuple

from langchain_groq import ChatGroq

from .prompts import RAG_PROMPT
from .vector_store import VectorStore


class RAGPipeline:

    def __init__(self):
        self.vector_store = VectorStore()

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

    def build_index(self, documents):
        self.vector_store.create(documents)

    def retrieve(
        self,
        question: str,
        k: int = 4
    ):
        return self.vector_store.search(
            question,
            k=k
        )

    def generate_answer(
        self,
        question: str,
        results
    ):

        if not results:
            return (
                "I couldn't find enough information "
                "in the provided documents.",
                []
            )

        context_parts = []
        sources = []

        for document, score in results:

            page = document.metadata.get("page", 0) + 1
            source = document.metadata.get(
                "source",
                "Unknown document"
            )

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n"
                f"{document.page_content}"
            )

            sources.append({
                "source": source,
                "page": page,
                "score": float(score)
            })

        context = "\n\n".join(context_parts)

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        response = self.llm.invoke(prompt)

        return response.content, sources