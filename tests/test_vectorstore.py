from pathlib import Path

from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


def test_ingest_and_list_documents(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    file_path = corpus / "ann_test.md"
    file_path.write_text(
        "# ANN\n\n## Basics\nArtificial neural networks contain neurons, weights, and activation functions. "
        "They learn by backpropagation and gradient-based updates over repeated training iterations. "
        "This makes them useful for nonlinear prediction tasks across many domains.",
        encoding="utf-8",
    )
    settings = Settings(chroma_dir=str(tmp_path / "chroma_db"), collection_name="test_collection")
    store = VectorStoreManager(settings)
    result = store.ingest(file_path)
    assert result.chunks_added >= 1
    assert "ann_test.md" in store.list_documents()


def test_duplicate_detection(tmp_path: Path):
    file_path = tmp_path / "cnn_test.md"
    file_path.write_text(
        "# CNN\n\n## Core\nConvolutional neural networks use filters and pooling to learn hierarchical visual features. "
        "They are especially strong for image classification because they preserve spatial structure.",
        encoding="utf-8",
    )
    settings = Settings(chroma_dir=str(tmp_path / "chroma_db2"), collection_name="test_collection_2")
    store = VectorStoreManager(settings)
    first = store.ingest(file_path)
    second = store.ingest(file_path)
    assert first.duplicate_skipped is False
    assert second.duplicate_skipped is True
