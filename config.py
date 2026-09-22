"""Configuracion compartida por la ingesta y la consulta."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
COLLECTION_NAME = "huerta_urbana"

# El mismo modelo se importa desde aqui al indexar y al consultar.
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

