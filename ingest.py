
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# 1. Load the Synk Corp text document.
file_path = Path("documents/synk_company_info.txt")

if not file_path.exists():
    raise FileNotFoundError(
        f"Document not found: {file_path}"
    )

documents = TextLoader(
    str(file_path),
    encoding="utf-8"
).load()

# 2. Split the document into smaller chunks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

# 3. Create embeddings locally using Hugging Face.
# OpenRouter is used for the chat model, not this step.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store document vectors in persistent local Qdrant storage.
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_data",
    collection_name="synk_company_docs",
    force_recreate=True
)

print(f"Indexed {len(chunks)} document chunks.")
print("Qdrant data saved in ./qdrant_data")
