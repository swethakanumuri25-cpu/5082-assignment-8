# Architecture

## End-to-end Flow
1. User uploads markdown or PDF files.
2. The chunker parses files into small topic-focused chunks.
3. Each chunk gets metadata such as topic, source, difficulty, and type.
4. The vector store embeds and stores chunks in ChromaDB.
5. A user query is embedded and matched against stored chunks.
6. LangGraph runs a small workflow:
   - retrieve context
   - check whether context exists
   - answer from context, or return a no-context guard message
7. The UI displays the answer and source citations.

## Why this works
- Good chunking improves retrieval precision.
- Metadata improves traceability.
- Grounded prompting reduces hallucination.
- A simple graph makes the behavior explainable in a demo.
