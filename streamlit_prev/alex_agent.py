import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from anthropic import Anthropic

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ============================================================
# Embedding Setup
# ============================================================

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


# ============================================================
# HonTe RAG Agent
# ============================================================

class HonTeAgent:
    def __init__(self):
        self.embedding = get_embedding()
        self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        self.max_tokens = int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "2048"))
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", "0"))

        base_dir = Path(__file__).resolve().parent
        index_path = base_dir / "honte_faiss_index"

        if not index_path.exists():
            raise FileNotFoundError(
                "HonTe FAISS index not found. "
                "Please upload documents via the admin page first."
            )

        self.vectorstore = FAISS.load_local(
            str(index_path),
            embeddings=self.embedding,
            allow_dangerous_deserialization=True,
        )

    def query(self, question: str, k: int = 20) -> Dict[str, Any]:
        # Retrieve relevant context
        docs = self.vectorstore.similarity_search(question, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        system_prompt = """You are an AI that deeply understands the thinking, philosophy, 
and mental models of HonTe's CEO based on their books, transcripts, and interviews.

When answering questions:
- Respond in the CEO's voice, tone, and style of reasoning
- Draw directly from their documented thinking and frameworks
- Reference specific ideas or concepts from their work where relevant
- Be direct, thoughtful, and substantive
- If the context doesn't contain enough information to answer confidently, say so

You are not roleplaying — you are synthesizing and reflecting their documented worldview."""

        user_prompt = f"""Question: {question}

Relevant context from CEO's books and transcripts:
{context}

Answer as if you are reflecting the CEO's perspective and thinking."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

        return {
            "answer": answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def build_agent():
    return HonTeAgent()