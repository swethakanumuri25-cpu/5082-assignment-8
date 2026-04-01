from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag_agent.agent.nodes import AgentService
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager

st.set_page_config(page_title="Deep Learning RAG Interview Prep Agent", layout="wide")


@st.cache_resource

def get_store() -> VectorStoreManager:
    return VectorStoreManager(Settings())


@st.cache_resource

def get_agent() -> AgentService:
    return AgentService(get_store(), Settings())


def initialise_session_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_generated_question", "")


def save_uploaded_file(uploaded_file) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "rag_agent_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def main() -> None:
    initialise_session_state()
    store = get_store()
    agent = get_agent()

    st.title("Deep Learning RAG Interview Prep Agent")
    left, middle, right = st.columns([1, 1, 1.2])

    with left:
        st.subheader("1. Ingestion")
        uploaded_files = st.file_uploader("Upload .md or .pdf files", type=["md", "pdf"], accept_multiple_files=True)
        if st.button("Ingest Uploaded Files", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                for uploaded in uploaded_files:
                    path = save_uploaded_file(uploaded)
                    result = store.ingest(path)
                    if result.duplicate_skipped:
                        st.info(f"{result.filename}: {result.message}")
                    else:
                        st.success(f"{result.filename}: {result.message}")

        if st.button("Load Sample Corpus", use_container_width=True):
            results = store.ingest_corpus_directory("data/corpus")
            for result in results:
                if result.duplicate_skipped:
                    st.info(f"{result.filename}: {result.message}")
                else:
                    st.success(f"{result.filename}: {result.message}")

        st.markdown("### Documents in Store")
        docs = store.list_documents()
        if docs:
            for name in docs:
                st.write(f"- {name}")
        else:
            st.caption("No documents ingested yet.")

    with middle:
        st.subheader("2. Document Viewer")
        docs = store.list_documents()
        selected = st.selectbox("Select document", ["-- Select --", *docs])
        if selected != "-- Select --":
            chunks = store.get_document_chunks(selected)
            st.write(f"Total chunks: {len(chunks)}")
            for idx, chunk in enumerate(chunks, start=1):
                with st.expander(f"Chunk {idx} | {chunk.metadata.get('topic', 'Unknown')}"):
                    st.write(chunk.text)
                    st.json(chunk.metadata)

    with right:
        st.subheader("3. Chat / Interview")
        question = st.text_input("Ask a deep learning question")
        if st.button("Get Answer", use_container_width=True):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                response = agent.answer_question(question)
                st.session_state.chat_history.append((question, response.answer, response.citations))

        for q, a, cites in reversed(st.session_state.chat_history):
            with st.container(border=True):
                st.markdown(f"**Question:** {q}")
                st.write(a)
                if cites:
                    st.caption("Citations: " + ", ".join(cites))

        st.markdown("### Generate Interview Question")
        topic_hint = st.text_input("Topic hint", value="LSTM")
        difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], index=1)
        if st.button("Generate Interview Question", use_container_width=True):
            data = agent.generate_question(topic_hint, difficulty)
            st.session_state.last_generated_question = data.get("question", "")
            st.json(data)

        st.markdown("### Evaluate Candidate Answer")
        eval_question = st.text_input("Question to evaluate", value=st.session_state.last_generated_question)
        candidate_answer = st.text_area("Candidate answer")
        if st.button("Evaluate Answer", use_container_width=True):
            if not eval_question.strip() or not candidate_answer.strip():
                st.warning("Enter both the question and candidate answer.")
            else:
                result = agent.evaluate_answer(eval_question, candidate_answer)
                st.write(f"Score: {result.score}/10")
                st.write(result.feedback)
                if result.missing_points:
                    st.write("Missing points:")
                    for point in result.missing_points:
                        st.write(f"- {point}")


if __name__ == "__main__":
    main()
