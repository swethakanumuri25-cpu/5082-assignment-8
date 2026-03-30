from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_agent.agent.state import IngestionResult, RetrievedChunk
from rag_agent.config import EmbeddingFactory, Settings
from rag_agent.corpus.chunker import DocumentChunker


class VectorStoreManager:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embeddings = EmbeddingFactory.create(self.settings)
        self.chunker = DocumentChunker()
        self._initialise()

    def _initialise(self) -> None:
        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.store = Chroma(
            client=self.client,
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.settings.chroma_path),
        )

    def _clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, list):
                if len(value) == 0:
                    continue
                cleaned[key] = ", ".join(str(item) for item in value)
            else:
                cleaned[key] = value

        return cleaned

    def check_duplicate(self, file_hash: str) -> bool:
        collection = self.client.get_or_create_collection(self.settings.collection_name)
        existing = collection.get(where={"file_hash": file_hash}, include=["metadatas"])
        return bool(existing and existing.get("ids"))

    def ingest(self, file_path: str | Path) -> IngestionResult:
        path = Path(file_path)
        chunks, file_hash = self.chunker.chunk_file(path)

        if self.check_duplicate(file_hash):
            return IngestionResult(
                filename=path.name,
                duplicate_skipped=True,
                message="Duplicate file skipped.",
            )

        docs: List[Document] = []
        for chunk in chunks:
            cleaned_metadata = self._clean_metadata(chunk.metadata)
            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata=cleaned_metadata,
                )
            )

        if docs:
            self.store.add_documents(docs)

        return IngestionResult(
            filename=path.name,
            chunks_added=len(docs),
            duplicate_skipped=False,
            message=f"Added {len(docs)} chunks.",
        )

    def query(self, question: str, k: int | None = None) -> List[RetrievedChunk]:
        k = k or self.settings.retrieval_k
        results = self.store.similarity_search_with_score(question, k=k)

        output: List[RetrievedChunk] = []
        for doc, score in results:
            output.append(
                RetrievedChunk(
                    text=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                )
            )
        return output

    def list_documents(self) -> List[str]:
        collection = self.client.get_or_create_collection(self.settings.collection_name)
        data = collection.get(include=["metadatas"])
        names = sorted(
            {
                metadata.get("source", "unknown")
                for metadata in data.get("metadatas", [])
                if metadata
            }
        )
        return names

    def get_document_chunks(self, filename: str) -> List[RetrievedChunk]:
        collection = self.client.get_or_create_collection(self.settings.collection_name)
        data = collection.get(where={"source": filename}, include=["documents", "metadatas"])

        chunks: List[RetrievedChunk] = []
        for text, metadata in zip(data.get("documents", []), data.get("metadatas", [])):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    metadata=metadata or {},
                )
            )
        return chunks

    def ingest_corpus_directory(self, directory: str | Path) -> List[IngestionResult]:
        path = Path(directory)
        results: List[IngestionResult] = []

        for file_path in sorted(path.glob("*")):
            if file_path.suffix.lower() not in {".md", ".pdf"}:
                continue
            results.append(self.ingest(file_path))

        return results