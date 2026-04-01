from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class IngestionResult(BaseModel):
    filename: str
    chunks_added: int = 0
    duplicate_skipped: bool = False
    message: str = ""


class AgentResponse(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)


class QAResult(BaseModel):
    score: int
    feedback: str
    missing_points: List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    question: str
    mode: str
    context: str
    chunks: List[RetrievedChunk]
    answer: str
    citations: List[str]
