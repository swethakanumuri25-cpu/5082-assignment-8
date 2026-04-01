from __future__ import annotations

import json
from typing import List

from rag_agent.agent.prompts import ANSWER_EVALUATION_PROMPT, QUESTION_GENERATION_PROMPT, SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, QAResult, RetrievedChunk
from rag_agent.config import LLMFactory, Settings
from rag_agent.vectorstore.store import VectorStoreManager


class AgentService:
    def __init__(self, store: VectorStoreManager | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.store = store or VectorStoreManager(self.settings)
        self.llm = LLMFactory.create(self.settings)

    def answer_question(self, question: str) -> AgentResponse:
        chunks = self.store.query(question)
        if not chunks:
            return AgentResponse(answer="No relevant context was found in the uploaded corpus.", citations=[], retrieved_chunks=[])

        context = self._build_context(chunks)
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        response = self.llm.invoke(prompt)
        answer = getattr(response, "content", str(response))
        citations = self._citations(chunks)
        if not citations:
            answer = "No relevant context was found in the uploaded corpus."
        return AgentResponse(answer=answer, citations=citations, retrieved_chunks=chunks)

    def generate_question(self, topic_hint: str = "deep learning", difficulty: str = "intermediate") -> dict:
        chunks = self.store.query(topic_hint)
        context = self._build_context(chunks)
        prompt = QUESTION_GENERATION_PROMPT.format(context=context, difficulty=difficulty)
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            return json.loads(content)
        except Exception:
            return {
                "question": f"Explain an important concept related to {topic_hint}.",
                "difficulty": difficulty,
                "topic": topic_hint,
                "model_answer": "Use the retrieved context to explain the concept clearly.",
                "follow_up": "What is one limitation or advantage of this method?",
                "source_citations": self._citations(chunks),
            }

    def evaluate_answer(self, question: str, candidate_answer: str) -> QAResult:
        chunks = self.store.query(question)
        context = self._build_context(chunks)
        prompt = ANSWER_EVALUATION_PROMPT.format(
            question=question,
            candidate_answer=candidate_answer,
            context=context,
        )
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        try:
            data = json.loads(content)
            return QAResult(**data)
        except Exception:
            return QAResult(
                score=5,
                feedback="The answer is partially correct but the JSON response could not be parsed cleanly.",
                missing_points=["Add more detail from the source chunk."],
            )

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        lines = []
        for chunk in chunks:
            topic = chunk.metadata.get("topic", "Unknown")
            source = chunk.metadata.get("source", "unknown")
            lines.append(f"[SOURCE: {topic} | {source}]\n{chunk.text}")
        return "\n\n".join(lines)

    def _citations(self, chunks: List[RetrievedChunk]) -> List[str]:
        return [f"[SOURCE: {c.metadata.get('topic', 'Unknown')} | {c.metadata.get('source', 'unknown')}]" for c in chunks]
