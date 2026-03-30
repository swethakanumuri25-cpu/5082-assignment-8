from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "deep_learning_corpus")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "3"))

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


class EmbeddingFactory:
    @staticmethod
    def create(settings: Settings):
        provider = settings.llm_provider.lower().strip()
        if provider == "ollama":
            return OllamaEmbeddings(model=settings.ollama_model, base_url=settings.ollama_base_url)
        return SentenceTransformerEmbeddings(model_name=settings.embedding_model)


class LLMFactory:
    @staticmethod
    def create(settings: Settings):
        provider = settings.llm_provider.lower().strip()
        if provider == "ollama":
            return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)
