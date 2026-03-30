# Deep Learning RAG Interview Prep Agent

A minimal working RAG-powered interview prep app for deep learning topics using Streamlit, ChromaDB, LangChain, and LangGraph.

## Features
- Upload `.md` and `.pdf` files
- Chunk and ingest documents into ChromaDB
- Duplicate detection by content hash
- Query the corpus and answer with citations
- Off-topic guard when no relevant context is found
- Generate interview questions from retrieved context
- Evaluate candidate answers against retrieved context

## Project Structure

```
deep-learning-rag-agent/
├── docs/
├── data/corpus/
├── examples/
├── src/rag_agent/
│   ├── agent/
│   ├── corpus/
│   ├── ui/
│   └── vectorstore/
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## Quick Start

### 1) Clone and enter project
```bash
git clone <your-repo-url>
cd deep-learning-rag-agent
```

### 2) Install UV
Mac/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3) Install dependencies
```bash
uv sync
```

### 4) Configure environment
```bash
cp .env.example .env
```
Open `.env` and add your API key if you use Groq.

### 5) Run the app
```bash
uv run streamlit run src/rag_agent/ui/app.py
```

### 6) Run tests
```bash
uv run pytest tests/ -v
```

## Recommended Demo Flow
1. Start the app
2. Upload `ann_intermediate.md`, `cnn_intermediate.md`, `rnn_intermediate.md`
3. Ask: `What is the difference between ANN and CNN?`
4. Ask off-topic: `Who was the first emperor of Rome?`
5. Generate interview question
6. Submit a weak answer and evaluate it

## Notes
- For a 2-hour build, keep the UI simple.
- Groq is the fastest option if you do not want local model setup.
- If you want full local inference, switch to Ollama in `.env`.
