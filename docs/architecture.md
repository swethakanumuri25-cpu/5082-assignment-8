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

# System Architecture  
## Team: Project group15  
## Date: 03/30/2026  

## Members and Roles:
- Corpus Architect: Katukuri Sri Ramya  
- Pipeline Engineer: Leela Kumar  
- UX Lead: Swetha Kanumuri  
- Prompt Engineer: Navya Baddam  
- QA Lead: Yogi Venkata Balaji Muthakani  

---
---

## Architecture Diagram

                               +----------------------+
                               |      User / Student  |
                               +----------+-----------+
                                          |
                                          v
                             +---------------------------+
                             |       Streamlit UI        |
                             |---------------------------|
                             | 1. Ingestion Panel        |
                             | 2. Document Viewer        |
                             | 3. Chat / Interview Panel |
                             +------------+--------------+
                                          |
                  +-----------------------+------------------------+
                  |                        |                       |
                  v                        v                       v
        +----------------+      +-------------------+     +------------------+
        | Upload / Load  |      | View Documents    |     | Ask Question /   |
        | Corpus Files   |      | and Chunks        |     | Generate / Eval  |
        +-------+--------+      +-------------------+     +---------+--------+
                |                                                      |
                v                                                      v
        +---------------------+                               +----------------------+
        | DocumentChunker     |                               | AgentService         |
        | - parse .md/.pdf    |                               | LangGraph-style flow |
        | - split into chunks |                               +----------+-----------+
        +----------+----------+                                          |
                   |                                                     v
                   v                                           +----------------------+
        +---------------------------+                          | Retrieval Node       |
        | Attach Metadata           |                          | - embed query        |
        | topic, difficulty, source |                          | - search vector DB   |
        | related_topics, is_bonus  |                          +----------+-----------+
        +-------------+-------------+                                     |
                      |                                                   v
                      v                                         +----------------------+
        +-----------------------------+                         | Hallucination Guard  |
        | Duplicate Detection         |                         | similarity threshold |
        | - file hash check           |                         | no relevant context? |
        +-------------+---------------+                         +----+-------------+---+
                      |                                                |             |
          duplicate?  | yes                                            | no          | yes
           ---------->+ skip ingestion                                 v             v
                      |                                      +----------------+   +----------------------+
                      | no                                   | Generation Node|   | Guard Response       |
                      v                                      | LLM + prompts  |   | "No relevant context"|
        +-----------------------------+                      +--------+-------+   +----------------------+
        | Embedding Model             |                               |
        | all-MiniLM-L6-v2            |                               v
        +-------------+---------------+                    +-----------------------------+
                      |                                    | Answer / Question / Eval    |
                      v                                    | with source citations       |
        +-----------------------------+                    +--------------+--------------+
        | ChromaDB Vector Store       |                                   |
        | Persistent local storage    |                                   v
        +-----------------------------+                         +--------------------------+
                                                               | Conversation Memory      |
                                                               | st.session_state         |
                                                               | chat_history, selection  |
                                                               +--------------------------+

## Component Descriptions
### Corpus Layer
Source files location: data/corpus/
File formats used:
.md files for the main prototype, with optional support for .pdf
Landmark papers ingested:
Rumelhart, Hinton & Williams (1986) — Backpropagation
LeCun et al. (1998) — LeNet
Elman (1990) — RNNs
### Chunking strategy:
We used concept-based chunking from markdown sections instead of only fixed-size splitting. Each chunk was kept around 100–300 words so it could represent one interview concept clearly without becoming too broad. This was chosen because smaller chunks improve retrieval precision and make source citations easier to explain.
### Metadata schema:

### Duplicate detection approach:
Duplicate detection is based on a content hash of the file, not the filename. A content hash is more reliable because two different filenames can contain the exact same document, while the same filename can also be edited and changed over time.
#### Corpus coverage:
 ANN
 CNN
 RNN
 LSTM
 Seq2Seq
 Autoencoder
 SOM (bonus)
 Boltzmann Machine (bonus)
 GAN (bonus)
### Vector Store Layer
Database: ChromaDB — PersistentClient
Local persistence path: data/chroma_db
Embedding model:
sentence-transformers/all-MiniLM-L6-v2 via sentence-transformers
Why this embedding model:
We chose this model because it is lightweight, free to run locally, and fast enough for a classroom prototype. It gives a good balance between semantic retrieval quality and speed without needing external embedding APIs.
Similarity metric:
Cosine similarity / default vector similarity in ChromaDB. This works well for semantic embeddings because it focuses on directional similarity rather than raw magnitude.
Retrieval k:
3 chunks per query. We chose 3 because it gives enough context for the model to answer while reducing irrelevant chunk noise.
Similarity threshold:
A simple relevance threshold / fallback logic was used for the hallucination guard. It was calibrated manually during testing rather than through a formal experiment.
Metadata filtering:
Basic metadata exists in the chunk store, but direct filtering by topic/difficulty is not fully exposed in the UI yet. The design supports adding those filters later.
### Agent Layer
Framework: LangGraph-inspired flow with agent service orchestration
Graph nodes:
Node	Responsibility
query_rewrite_node	Optionally reformats or clarifies the user’s input before retrieval
retrieval_node	Sends the user query to the vector store and retrieves top matching chunks
generation_node	Builds the prompt with retrieved chunks and generates the final grounded answer
Conditional edges:
If no relevant chunks are found, the graph routes to the hallucination guard response instead of answer generation. If relevant context exists, it routes to the generation node.
Hallucination guard:
When relevant retrieval is not strong enough, the system returns a message like:
“I could not find enough relevant deep learning context in the current corpus to answer that reliably.”
Query rewriting:
Raw query: What is special about CNN?
Rewritten query: Explain what makes convolutional neural networks effective for image processing
Conversation memory:
Conversation memory is maintained through st.session_state, which stores chat history and document selection across Streamlit reruns. If the app restarts, this memory is lost because it is session-based, not database-backed.
LLM provider:
Groq with llama-3.1-8b-instant
Why this provider:
Groq was chosen because it is fast, easy to set up, and offers free access suitable for a hackathon or classroom prototype.
### Prompt Layer
System prompt summary:
The agent is instructed to behave like a grounded interview-prep assistant. It must answer only from retrieved context, cite sources clearly, and avoid unsupported claims.
Question generation prompt:
It takes retrieved context and a difficulty level as input. It returns a structured interview question, a model answer, and optionally a follow-up question.
Answer evaluation prompt:
It compares the candidate’s answer against the source chunk and question, then scores the response on understanding, completeness, correctness, and specificity.
JSON reliability:
We added instructions such as:
“Respond with the JSON object only. No preamble, explanation, or markdown code fences.”
Failure modes identified:
Question generation sometimes returned prose instead of JSON, so stricter output instructions were added
Answer evaluation sometimes scored vague answers too generously, so scoring criteria were clarified
System prompt could over-answer from model memory, so the grounding rule was made more explicit

### Interface Layer
Framework: Streamlit
Deployment platform: Streamlit Community Cloud
Public URL: (paste your deployed app URL here once live)
Ingestion panel features:
The user can upload .md or .pdf files, ingest them into the vector store, and also load a sample corpus. The panel also shows status messages like chunks added or duplicate skipped.
Document viewer features:
Users can select any ingested document, see how many chunks it contains, and expand each chunk to inspect the stored text and metadata.
Chat panel features:
The user can ask deep learning questions and receive grounded answers with source citations. The interface is also designed to surface a guarded response when there is not enough relevant context.

### Session state keys:
Key	Stores
chat_history	Previous user questions and system answers
ingested_documents	List of loaded document names
selected_document	Currently selected file in the viewer
thread_id	Identifier for the current interaction flow or session

### Stretch features implemented:

Interview question generation
Candidate answer evaluation
Expandable chunk viewer
Duplicate ingestion checking

### Design Decisions
Decision: Use concept-based chunks of roughly 100–300 words
Rationale:
This size keeps each chunk focused on a single interview concept while still providing enough information to answer meaningfully. Larger chunks would reduce retrieval precision, while very tiny chunks would lose context.
Interview answer:
We intentionally kept chunks small and concept-focused because retrieval quality depends on semantic precision. If the chunks were much larger, the model would receive noisier context and source citations would become less meaningful.
Decision: Use all-MiniLM-L6-v2 for embeddings
Rationale:
The model is lightweight, fast, and works locally without external API cost. This made it practical for a student project with limited setup time.
Interview answer:
We chose MiniLM because it offers a strong speed-to-quality tradeoff for semantic retrieval. It’s not the most powerful embedding model available, but it let us build a fast local retrieval system with low friction.
Decision: Use Groq with Llama 3.1 8B Instant as the LLM
Rationale:
Groq gave us fast inference and simple integration, which mattered more than maximum model size in a short project window.
Interview answer:
We prioritized deployment speed and response latency, so Groq was a practical choice. For this prototype, quick iteration mattered more than using the largest possible model.
Decision: Use content hash for duplicate detection
Rationale:
File names are unreliable because users can rename identical content. A content hash is stable for the actual file contents and better supports true duplicate detection.
Interview answer:
We used a content-based duplicate check because filename-based matching is brittle. Two different filenames can still represent the same document, so hashing gives a more robust signal.

### QA Test Results
Test	Expected	Actual	Pass / Fail
Normal query	Relevant chunks, source cited	Returned grounded ANN/CNN answer with citation	Pass
Off-topic query	No context found message	Needs final validation in live demo	Partial
Duplicate ingestion	Second upload skipped	Logic implemented, should be retested in final run	Partial
Empty query	Graceful error, no crash	No crash expected, should be validated	Partial
Cross-topic query	Multi-topic retrieval	ANN/CNN query retrieved relevant content	Pass

### Critical failures fixed before Hour 3:

Package import / rag_agent path issue
ChromaDB metadata error caused by empty related_topics list
Groq API key placeholder causing authentication failure

### Known issues not fixed (and why):

Off-topic threshold tuning is still manual because of time limits
Deployment and public URL may still need final verification

### Known Limitations
PDF chunking can still be noisy if a PDF has references or irregular formatting
Similarity threshold for the hallucination guard was calibrated manually, not through a benchmark
Conversation memory is stored only in session state and is lost if the app restarts
Metadata filtering is not fully exposed in the UI yet
Current corpus coverage is limited to the main three topics

### What We Would Do With More Time
Add LSTM, Seq2Seq, Autoencoder, and GAN content to the corpus
Implement hybrid retrieval with both semantic and keyword search
Add a re-ranking stage using a cross-encoder
Improve hallucination guard using a calibrated similarity threshold
Add persistent conversation memory across sessions
Support asynchronous ingestion for larger PDF files

### Hour 3 Interview Questions

Question 1:
Why is chunk size important in a RAG system?

Model answer:
Chunk size directly affects retrieval precision and answer quality. If chunks are too large, retrieval becomes noisy; if they are too small, they may lose the context needed for meaningful answers.

Question 2:
Why is a content hash better than a filename for duplicate detection?

Model answer:
A filename only identifies what the file is called, not what it contains. A content hash is based on the actual file data, so it can detect duplicates even when filenames differ.

Question 3:
Why do we use an embedding model separately from the LLM in this system?

Model answer:
The embedding model converts documents and queries into vectors for semantic similarity search, while the LLM generates natural language answers. Separating these roles improves efficiency and makes retrieval scalable.

### Team Retrospective

What clicked:

The corpus-to-vector-store-to-answer pipeline became much clearer once the sample corpus was ingested successfully
Streamlit made it easy to demonstrate the full workflow quickly

What confused us:

Environment setup and import paths caused initial delay
ChromaDB metadata formatting rules were stricter than expected
API key configuration caused avoidable troubleshooting time

One thing each team member would study before a real interview:

Corpus Architect: better chunking strategies and metadata design
Pipeline Engineer: LangGraph orchestration and vector database internals
UX Lead: Streamlit state management and deployment
Prompt Engineer: JSON-safe prompt engineering and hallucination prevention
QA Lead: retrieval evaluation metrics and guardrail testing
