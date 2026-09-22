"""Cadena RAG asincrona con recuperacion, respuesta grounded y salida tipada."""

import asyncio
import sys

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from config import (
    CHAT_MODEL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    VECTORSTORE_DIR,
)


class RagResponse(BaseModel):
    answer: str = Field(description="Respuesta en espanol o exactamente 'No lo sé'.")
    references: list[str] = Field(
        description="Archivos fuente usados; lista vacia si la respuesta es 'No lo sé'."
    )


def format_documents(documents: list[Document]) -> str:
    """Convierte los resultados del retriever en contexto con fuentes visibles."""
    return "\n\n".join(
        f"FUENTE: {doc.metadata['source']}\nFRAGMENTO: {doc.page_content}"
        for doc in documents
    )


def build_rag_chain():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )
    if vectorstore._collection.count() == 0:
        raise RuntimeError("La base vectorial esta vacia. Ejecuta primero: python ingest.py")

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    parser = PydanticOutputParser(pydantic_object=RagResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente sobre huerta urbana. Responde exclusivamente con datos "
                "del CONTEXTO. No uses conocimiento externo ni completes informacion. Si el "
                "contexto no permite responder, answer debe ser exactamente 'No lo sé' y "
                "references debe ser una lista vacia. Si respondes, incluye en references solo "
                "los nombres de las FUENTES que realmente utilizaste.\n\n"
                "{format_instructions}",
            ),
            ("human", "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # LCEL: pregunta -> retriever -> formateo -> prompt -> LLM -> Pydantic.
    return (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
        | prompt
        | ChatOpenAI(model=CHAT_MODEL, temperature=0)
        | parser
    )


async def get_rag_response(query: str) -> RagResponse:
    """Ejecuta de manera asincrona el flujo RAG completo."""
    if not query.strip():
        raise ValueError("La consulta no puede estar vacia.")
    return await build_rag_chain().ainvoke(query)


async def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or input("Pregunta: ").strip()
    result = await get_rag_response(query)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
