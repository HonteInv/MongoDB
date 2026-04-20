import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    TextLoader,
)

load_dotenv()

# ============================================================
# Embeddings
# ============================================================

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

if EMBEDDING_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif EMBEDDING_PROVIDER == "gemini":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"   
INDEX_PATH = BASE_DIR / "honte_faiss_index"

# ============================================================
# Loaders
# ============================================================

def load_documents(folder: Path):
    documents = []
    for file in folder.rglob("*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() == ".md":
            loader = UnstructuredMarkdownLoader(str(file))
        elif file.suffix.lower() == ".csv":
            loader = CSVLoader(str(file))
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file))
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded: {file.name} ({len(docs)} docs)")
    return documents

# ============================================================
# Build Index
# ============================================================

def build_index():
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        print("Create a 'data/' folder and add your PDF/MD/TXT/CSV files.")
        return

    print(f"Loading documents from: {DATA_DIR}")
    docs = load_documents(DATA_DIR)

    if not docs:
        print("No documents found in data/ folder.")
        return

    print(f"\nChunking {len(docs)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("\nBuilding FAISS index...")
    if INDEX_PATH.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_PATH),
            embeddings=embedding,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
        print("Updated existing index.")
    else:
        vectorstore = FAISS.from_documents(chunks, embedding)
        print("Created new index.")

    vectorstore.save_local(str(INDEX_PATH))
    print(f"\nIndex saved to: {INDEX_PATH}")
    print("Done. Commit honte_faiss_index/ to git to deploy.")

if __name__ == "__main__":
    build_index()