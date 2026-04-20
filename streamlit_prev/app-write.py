import streamlit as st
import os
import csv
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

st.set_page_config(
    page_title="RAG Admin",
    page_icon=None,
    layout="wide"
)

# -------------------------
# Top Navigation Bar
# -------------------------

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
        header[data-testid="stHeader"] { display: none !important; }

        .navbar {
            display: flex;
            align-items: center;
            background-color: #f5f0e8;
            padding: 14px 32px;
            margin: -60px -4rem 32px -4rem;
            border-bottom: 1px solid #ddd5c4;
            gap: 32px;
        }
        .navbar-brand {
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            font-size: 17px;
            color: #2c2c2c;
            margin-right: auto;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .navbar a {
            font-family: 'Cormorant Garamond', serif;
            color: #7a6e60;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            transition: color 0.2s;
        }
        .navbar a:hover { color: #2c2c2c; }
    </style>

    <div class="navbar">
        <span class="navbar-brand">Navigation</span>
        <a href="https://honte-search-app.streamlit.app/" target="_blank">Search</a>
        <a href="https://honte-pnl-query.streamlit.app/" target="_blank">PnL Query</a>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Jost:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Jost', sans-serif;
        background-color: #ffffff;
        color: #2c2c2c;
    }

    .main { background-color: #ffffff; }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }

    p, .stMarkdown p {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #4a4a4a !important;
        font-size: 15px !important;
    }

    /* Password input */
    .stTextInput label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stTextInput input {
        background-color: #faf8f5 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-size: 15px !important;
        font-weight: 300 !important;
        box-shadow: none !important;
    }

    .stTextInput input:focus {
        border-color: #b8a99a !important;
        box-shadow: 0 0 0 1px #b8a99a !important;
    }

    /* File uploader */
    .stFileUploader label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stFileUploader > div {
        background-color: #faf8f5 !important;
        border: 1px dashed #ddd5c4 !important;
        border-radius: 2px !important;
    }

    /* Selectbox */
    .stSelectbox label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stSelectbox > div > div {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #2c2c2c !important;
    }

    /* Button */
    .stButton > button {
        background-color: #2c2c2c !important;
        color: #f5f0e8 !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 400 !important;
        font-size: 13px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 1px !important;
        padding: 0.55rem 2.2rem !important;
        transition: background-color 0.2s !important;
    }

    .stButton > button p {
        color: #f5f0e8 !important;
    }

    .stButton > button:hover { background-color: #4a4a4a !important; }

    /* Alerts */
    .stSuccess {
        background-color: #f5f5f0 !important;
        border: 1px solid #c9a87a !important;
        border-left: 3px solid #c9a87a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stInfo {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-left: 3px solid #b8a99a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stWarning {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    .stError {
        background-color: #fdf5f5 !important;
        border: 1px solid #e8c4c4 !important;
        color: #8a4a4a !important;
        border-radius: 1px !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #c9a87a !important; }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid #e8e2d9 !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Password Protection
# ============================================================

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

password = st.text_input("Admin Password", type="password")

if password != ADMIN_PASSWORD:
    st.stop() 

# ============================================================
# Embedding Setup (Cached)
# ============================================================

@st.cache_resource
def get_embedding():
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

    if provider == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    else:
        raise ValueError("Unsupported embedding provider.")

embedding = get_embedding()


# ============================================================
# File Loaders
# ============================================================

def load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    return loader.load()

def load_md(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name})]

def load_csv(path: Path) -> list[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            docs.append(Document(page_content=content, metadata={"source": path.name, "row": i}))
    return docs

def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".md":
        return load_md(path)
    elif ext == ".csv":
        return load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ============================================================
# UI
# ============================================================

st.title("Upload & Index Documents")

uploaded_files = st.file_uploader("Upload Doc (pdf, markdown, CSV)", 
                                 type=["pdf", "md", "csv"], 
                                 accept_multiple_files=True
                                 )

store_type = st.selectbox(
    "Select Knowledge Base",
    ["context", "pnl", "newsletter", "weekly_market_data"]
)

if uploaded_files and st.button("Index Document"):

    BASE_DIR = Path(__file__).resolve().parent
    index_path = BASE_DIR / f"{store_type}_faiss_index"

    all_chunks = []
    errors = []

    with st.spinner("Processing and embedding document..."):

        for uploaded_file in uploaded_files: 
            temp_path = BASE_DIR / uploaded_file.name
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                docs = load_file(temp_path) 

                # Chunk
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                st.write(f"{uploaded_file.name}` — {len(chunks)} chunks")

            except Exception as e:
                errors.append(f"{uploaded_file.name}: {str(e)}")
                st.warning(f"Error processing {uploaded_file.name}: {str(e)}") 

            finally:
                if temp_path.exists():
                    os.remove(temp_path)

        if all_chunks:
            if index_path.exists():
                vectorstore = FAISS.load_local(
                    str(index_path),
                    embeddings=embedding,
                    allow_dangerous_deserialization=True,
                )
                vectorstore.add_documents(all_chunks)
                st.info("Updated existing index.")
            else:
                vectorstore = FAISS.from_documents(chunks, embedding)
                st.info("Created new index.")

            vectorstore.save_local(str(index_path))

    if all_chunks:
        st.success(f"Document indexed successfully! {len(all_chunks)} total chunks indexed into `{store_type}`.")
    if errors:
        st.error(f"Failed to process: {', '.join(errors)}")
