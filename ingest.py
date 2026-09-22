"""Lee documentos locales, los fragmenta y los persiste en ChromaDB."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)


def load_documents() -> list[Document]:
    """Carga los .txt y .md de data, conservando el nombre como metadata."""
    paths = sorted([*DATA_DIR.glob("*.txt"), *DATA_DIR.glob("*.md")])
    if not paths:
        raise FileNotFoundError(f"No se encontraron archivos .txt o .md en {DATA_DIR}")

    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in paths
    ]


def ingest_documents() -> None:
    """Crea el indice una sola vez; evita embeddings repetidos si ya tiene datos."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    existing_chunks = vectorstore._collection.count()
    if existing_chunks > 0:
        print(f"La coleccion ya contiene {existing_chunks} fragmentos. No se reindexo.")
        return

    # from_tiktoken_encoder mide el tamano y el solapamiento en tokens.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(load_documents())
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index

    vectorstore.add_documents(chunks)
    print(f"Se indexaron {len(chunks)} fragmentos en {VECTORSTORE_DIR}.")


if __name__ == "__main__":
    ingest_documents()

