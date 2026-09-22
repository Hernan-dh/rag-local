# rag-local

Proyecto de recuperacion semantica sobre tres documentos de huerta urbana. Los documentos y la base vectorial son locales; OpenAI se usa para crear embeddings y generar la respuesta.

## Requisitos

- Python 3.11 o 3.12
- Una API key de OpenAI

## Instalacion

```bash
python -m venv .venv
```

En Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

En Linux o macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edita `.env` y coloca tu clave en `OPENAI_API_KEY`. Este archivo esta ignorado por Git.

## Uso

Primero crea la base vectorial persistente:

```bash
python ingest.py
```

Si la coleccion ya tiene datos, el script no vuelve a indexarlos. Para reconstruirla despues de modificar el dataset, elimina manualmente la carpeta `vectorstore` y ejecuta la ingesta otra vez.

Luego realiza una consulta:

```bash
python rag.py "¿Cuantas horas de sol necesita un tomate?"
```

La salida es JSON validado con Pydantic e incluye `answer` y `references`.

## Pruebas solicitadas

Despues de la ingesta, ejecuta:

```bash
python test_queries.py
```

El script comprueba dos casos:

1. Una pregunta respondida por los documentos, que debe devolver texto y referencias.
2. Una pregunta sobre el Mundial de 1998, ausente del dataset, que debe devolver `No lo sé` y ninguna referencia.

Las pruebas llaman a OpenAI y por eso requieren conexion y saldo disponible.

## Estructura

```text
.
|-- data/
|   |-- luz_y_cultivos.txt
|   |-- riego.txt
|   `-- sustrato.md
|-- config.py
|-- ingest.py
|-- rag.py
|-- test_queries.py
|-- requirements.txt
|-- .env.example
`-- README.md
```

`ingest.py` usa `RecursiveCharacterTextSplitter` con fragmentos de 500 tokens y 50 tokens de solapamiento. `rag.py` recupera como maximo cuatro fragmentos y compone la cadena con LCEL. El mismo modelo de embeddings configurado en `config.py` se usa en ambos pasos.
