from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

from rag_agent.agent.state import RetrievedChunk


def read_text_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def file_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


class DocumentChunker:
    def chunk_file(self, file_path: Path) -> Tuple[List[RetrievedChunk], str]:
        raw_text = read_text_file(file_path)
        digest = file_hash(raw_text)
        topic = self._infer_topic(file_path, raw_text)
        sections = self._split_sections(raw_text)
        chunks: List[RetrievedChunk] = []

        for idx, section in enumerate(sections, start=1):
            clean = self._clean_text(section)
            if len(clean.split()) < 40:
                continue
            metadata = {
                "topic": topic,
                "difficulty": "intermediate",
                "type": "concept_explanation",
                "source": file_path.name,
                "chunk_id": f"{file_path.stem}_{idx}",
                "file_hash": digest,
                "related_topics": [],
                "is_bonus": False,
            }
            chunks.append(RetrievedChunk(text=clean, metadata=metadata))

        return chunks, digest

    def _infer_topic(self, file_path: Path, text: str) -> str:
        name = file_path.stem.lower()
        if "ann" in name:
            return "ANN"
        if "cnn" in name:
            return "CNN"
        if "rnn" in name or "lstm" in name:
            return "RNN"
        first_heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        return first_heading.group(1).strip() if first_heading else file_path.stem

    def _split_sections(self, text: str) -> List[str]:
        parts = re.split(r"\n##+\s+", text)
        if len(parts) == 1:
            return [p.strip() for p in text.split("\n\n") if p.strip()]
        return [p.strip() for p in parts if p.strip()]

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
