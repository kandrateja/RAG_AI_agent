# RAG AI Agent

A comprehensive Retrieval-Augmented Generation (RAG) AI agent that combines Azure Document Intelligence for OCR, Azure OpenAI for embeddings and LLM, Postgres+pgvector for vector storage, Neo4j for graph storage, and a Surf-style web search API.

## 🚀 Quick Start

**New to the project?** See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for complete step-by-step setup and testing instructions.

## Features

- **Azure Document Intelligence**: OCR and page-accurate text extraction from scanned PDFs (non-text, image-based)
- **Azure OpenAI**: 
  - Text embeddings using `text-embedding-3-small` (1536 dims, HNSW compatible) 
  - LLM chat completions using Azure OpenAI deployments (e.g., `gpt-4o`)
  - Vision captions (page-level) via a vision-capable deployment (e.g., `gpt-4o`)
  - Entity + relationship extraction (LLM-assisted) to populate the graph
- **Vector DB (Postgres + pgvector)**:
  - Primary semantic retrieval over chunks
  - **Hybrid search**: Combines semantic similarity (70%) + flexible keyword search (30%)
  - **Flexible keyword search**: Uses PostgreSQL full-text search with OR logic (any query word can match)
  - HNSW index enabled for fast similarity search (when embedding dims <= 2000)
  - GIN index on text column for fast keyword search
  - Stores required provenance per chunk: **doc_id, page_number, chunk_id, text**
- **Graph DB (Neo4j)**:
  - Stores entities + relationships
  - Linked to chunk_ids so graph is **used meaningfully at query time**
- **Web Search (Surf-like API)**:
  - External web search using a Surf-style HTTP API
  - **Fallback only** when internal knowledge is insufficient
- **Document Deduplication**: Prevents re-ingesting the same document
- **Comprehensive Citations**: All answers include citations with doc_id, page_number, and chunk_id
- **Provenance Tracking**: Explicit indication of answer source (internal/online/both/none)
- **Decision Trace**: Shows routing thresholds, similarity scores, and graph confidence
- **Re-ranking**: Initial fetch → rerank (vector + keyword + graph boost) → final top_k
- **Robust Error Handling**: Handles OCR failures, empty retrieval, tool failures gracefully
- **Observability**: Comprehensive logging of all retrieval steps and tool calls

## Project Structure

```
RAG_AI_agent/
├── src/
│   ├── ocr/
│   │   └── document_processor.py      # Azure Document Intelligence OCR
│   ├── embeddings/
│   │   └── embedding_generator.py     # Azure OpenAI embeddings
│   ├── llm/
│   │   └── azure_openai_client.py    # Azure OpenAI LLM client
│   ├── graphrag/
│   │   ├── neo4j_client.py           # Neo4j client with GraphRAG
│   │   └── text_chunker.py           # Text chunking utilities
│   ├── websearch/
│   │   └── surf_client.py            # Surf-like web search client
│   ├── ner/
│   │   └── entity_extractor.py      # LLM-based entity & relationship extraction
│   ├── vectorstore/
│   │   └── postgres_pgvector.py     # Postgres + pgvector vector DB
│   ├── api/
│   │   └── server.py                 # FastAPI server for HTTP access
│   └── rag_agent.py                  # Main RAG agent orchestrator & routing
├── config.py                          # Configuration settings
├── main.py                            # CLI entry point
├── example_usage.py                   # Usage examples
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
└── README.md                          # This file
```

## Prerequisites

1. **Azure Account** with:
   - Azure Document Intelligence resource
   - Azure OpenAI resource with:
     - Chat completion model deployment (e.g., GPT-4o)
     - Embedding model deployment (`text-embedding-3-small` recommended for HNSW)

2. **Docker** and **Docker Compose** (for running databases and optionally the entire system)

3. **Python 3.9+** (if running the app locally instead of in Docker)

## Quick Start with Docker Compose (Recommended)

The easiest way to run the entire system:

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RAG_AI_agent
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and fill in your Azure credentials
```

3. **Update database connection strings in .env** (for Docker Compose):
```env
POSTGRES_DSN=postgresql://postgres:postgres@postgres:5432/rag
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=rag-neo4j-password-2024
```

**Important**: The Docker Compose file sets Neo4j password to `rag-neo4j-password-2024`. Make sure your `.env` file matches this, or update `docker-compose.yml` with your preferred password.

4. **Start everything with Docker Compose**:
```bash
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector (port 5432)
- Neo4j (ports 7474, 7687)
- RAG API server (port 8000)

5. **Access the UI**: Open `http://localhost:8000/`

6. **Stop everything**:
```bash
docker-compose down
```

**Note**: For local development, you can run only databases with Docker and run the Python app locally:
```bash
# Start only databases
docker-compose -f docker-compose.dev.yml up -d

# Then run Python app locally
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

## Manual Installation (Alternative)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd RAG_AI_agent
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-document-intelligence-key

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j

# Postgres (pgvector)
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag

# Web Search (Surf-like API)
SURF_API_ENDPOINT=https://your-surf-endpoint.example.com/search
SURF_API_KEY=your-surf-api-key
SURF_MAX_RESULTS=5
```

## Usage

### Option 1: Docker Compose (Recommended)

If you used Docker Compose, everything is already running:
- UI: `http://localhost:8000/`
- API: `http://localhost:8000/api/...`
- Neo4j Browser: `http://localhost:7474/`

### Option 2: Manual Setup

### Recommended: UI

1. Start the API server:
```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```
2. Open `http://localhost:8000/`, ingest a PDF, and chat.

### Optional: Python Usage

#### 1. Ingest a Document

```python
from src.rag_agent import RAGAgent

agent = RAGAgent()
result = agent.ingest_document("path/to/document.pdf")
print(f"Document ID: {result['doc_id']}")
print(f"Chunks created: {result['chunks_created']}")
agent.close()
```

#### 2. Query the System

```python
from src.rag_agent import RAGAgent

agent = RAGAgent()
result = agent.query("What are muscle spindles, and how do they contribute to proprioception?", top_k=5)
print(f"Answer: {result['answer']}")
print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
agent.close()
```

### Optional: HTTP API

#### Endpoints

- `GET /health` – simple health check
- `POST /ingest` – ingest a document via file upload
  - Form-data:
    - `file`: uploaded file
    - `doc_id` (optional): custom document ID
- `POST /query` – query the RAG system
  - JSON body:
    - `question`: string
    - `top_k`: integer (default 5)
    - `use_graph_context`: boolean (default `true`)

Example query request:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What are muscle spindles, and how do they contribute to proprioception?
",
        "top_k": 5,
        "use_graph_context": true
      }'
```

Example query response:

```json
{
  "question": "What are muscle spindles, and how do they contribute to proprioception?
",
  "answer": "Based on the internal knowledge base...",
  "citations": [
    {
      "chunk_id": "doc123_chunk_0",
      "doc_id": "doc123",
      "page_number": 1,
      "similarity": 0.85
    }
  ],
  "provenance": "internal",
  "tools_used": {
    "graph": true,
    "vector": false,
    "web": false
  },
  "retrieved_chunks_count": 5,
  "web_results_count": 0,
  "has_internal_knowledge": true,
  "internal_sufficient": true
}
```


```python
from src.rag_agent import RAGAgent

agent = RAGAgent()
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = agent.ingest_batch(documents)
agent.close()
```

## Architecture

### Document Ingestion Flow

1. **Deduplication Check**: Uses `doc_id` and PDF `doc_hash` to prevent duplicates.
2. **OCR Extraction**: Azure Document Intelligence extracts page-level text from scanned PDFs.
3. **Vision Captions (page-level)**: Each PDF page is rendered to an image and sent to a vision-capable model (e.g., `gpt-4o`) to detect whether a diagram/table is present. Captions are only kept when a diagram is detected, and are merged into the same page text to stay aligned by `page_number`.
4. **Entity/Relationship Extraction**: LLM-based extraction creates entity/relationship nodes in Neo4j.
5. **Text Chunking**: Split into chunks, preserving `page_number`.
6. **Embedding Generation**: Azure OpenAI generates embeddings for each chunk.
7. **Vector Storage (Postgres + pgvector)**: Stores chunk embeddings + required provenance (`doc_id`, `page_number`, `chunk_id`, `text`).
8. **Graph Storage (Neo4j)**: Stores entities/relationships, plus `ChunkRef(chunk_id)` nodes linked via `(:ChunkRef)-[:MENTIONS]->(:Entity)` for query-time graph expansion.

### Query Flow

1. **Vector Retrieval (pgvector)**: Fetch initial_k, rerank, then keep top_k chunks.
   - **Hybrid search**: Combines semantic similarity (cosine) + flexible keyword search (PostgreSQL full-text with OR logic)
   - **Flexible keyword matching**: Query words are joined with OR (`word1 | word2 | word3`), so chunks matching any word are included (ranked by relevance)
   - **Re-ranking**: Combines hybrid similarity (60%) + keyword overlap (25%) + graph boost (15%)
   - Graph boost favors chunks linked to entities in Neo4j.
2. **Routing by thresholds**:
   - `vector_best_score >= 0.7` → vector-only
   - `0.3 <= vector_best_score < 0.7` → vector + graph
   - `vector_best_score < 0.3` → web fallback
3. **Graph Expansion (Neo4j)**:
   - Extract entities from the question; if none, extract from top vector chunks
   - Use chunk-linked entities (`ChunkRef -> Entity`) to expand context
4. **Graph Confidence (Neo4j)**:
   - `coverage`: % of entities found in graph
   - `path_score`: connectivity between entities (shortest paths)
   - `relation_score`: strength/volume of relationships
   - `confidence`: weighted aggregate used for sufficiency
   - If `confidence < graph_confidence_threshold` (default `0.6`), the system falls back to web search
5. **Answer + Provenance**:
   - If `vector_best_score < 0.3`, internal citations are cleared and only web citations are shown
   - Responses always include provenance + sources used (vector/graph/web/direct)

## Configuration

Edit `config.py` or set environment variables to customize:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `max_tokens`: Maximum tokens for LLM responses (default: 4096)
- Routing thresholds and graph confidence weights live in `src/rag_agent.py` and `src/graphrag/graph_confidence.py`

## Components

### Document Processor (`src/ocr/document_processor.py`)
- Uses Azure Document Intelligence API
- Supports various document formats (PDF, images, etc.)
- Extracts page-level text using OCR lines
- Renders page images for vision captioning (PyMuPDF)
- Captions are only kept when a diagram is detected, then merged into page text before chunking

### Embedding Generator (`src/embeddings/embedding_generator.py`)
- Uses Azure OpenAI embedding models
- Supports `text-embedding-3-small` (1536 dimensions)
- Batch processing support

### Postgres Vector Store (`src/vectorstore/postgres_pgvector.py`)
- Implements hybrid semantic + keyword search
- **Semantic search**: Cosine similarity using pgvector (HNSW index when dims <= 2000)
- **Keyword search**: PostgreSQL full-text search with flexible OR-based matching
  - Query words joined with OR (`word1 | word2 | word3`)
  - Uses `to_tsquery` for flexible matching and `ts_rank_cd` for ranking
  - GIN index on `to_tsvector('english', text)` for performance
- Hybrid scoring: 70% semantic + 30% keyword (normalized)
- Stores chunk embeddings with provenance (doc_id, page_number, chunk_id, text)

### Neo4j Client (`src/graphrag/neo4j_client.py`)
- Chunk/entity linking and graph expansion
- Entity and relationship management

### Graph Confidence (`src/graphrag/graph_confidence.py`)
- Scores graph results using coverage, path score, and relationship strength
- Produces a `confidence` value used in routing

### Web Search Client (`src/websearch/surf_client.py`)
- Thin wrapper over a Surf-style HTTP web search endpoint
- Normalizes results into title / URL / snippet triples
- Used when the router decides that open-web information is needed

### RAG Agent (`src/rag_agent.py`)
- Orchestrates ingestion + retrieval
- Applies vector/graph/web routing rules
- Generates citations and provenance
- Emits `decision_trace` (thresholds + scores) and `sources_used`
- Logs decisions and tool usage

### Entity Extractor (`src/ner/entity_extractor.py`)
- Extracts named entities (Person, Organization, Location, Concept, etc.)
- Extracts relationships between entities
- Uses Azure OpenAI for extraction
- Stores entities and relationships in Neo4j graph
- No separate “Sanity extractor” is used

## Neo4j Schema

The system creates the following node types and relationships:

- **Document**: Represents ingested documents (with doc_id, content, doc_hash for deduplication)
- **ChunkRef**: Chunk references linked to entities
- **Entity**: Extracted entities with type and name
- **MENTIONS**: ChunkRef → Entity relationship
- **Entity relationships**: Entity → Entity links (type varies)

## Requirements Compliance

This system meets all the functional requirements:

✅ **Document Ingestion Pipeline**
- Supports scanned PDFs (non-text, image-based) via Azure Document Intelligence OCR
- Text preprocessing and chunking with page number preservation
- Embedding generation using Azure OpenAI
- Storage in Postgres (vector) and Neo4j (graph)
- Entity and relationship extraction
- Each chunk retains: doc_id, page_number, chunk_id, text_content
- Re-ingesting same document is prevented (deduplication)

✅ **Hybrid Knowledge Base**
- Vector index for semantic similarity search (Postgres + pgvector)
- Graph database representing entities and relationships (Neo4j)
- Graph is used meaningfully at query time (chunk-linked expansion)

✅ **Query & Agent Interface**
- FastAPI chat/API interface
- Answers using internal knowledge base first
- Tool calling supports: knowledge base search, graph queries, online search
- Falls back to online search only when internal knowledge is insufficient
- Routing decisions are surfaced via `decision_trace` (thresholds + scores)

✅ **Answer Format & Provenance**
- Clear, concise answers
- Citations with: Document ID, Page number, Chunk ID
- Explicit indication of source: internal/online/both/none
- Clear statement when answer cannot be found (no hallucination)

✅ **Observability & Reliability**
- Comprehensive logging of retrieval steps and tool calls
- Error handling for:
  - OCR failures (returns error_type: "ocr_failure")
  - Empty retrieval results (returns provenance: "none")
  - Tool failures (graceful fallback and error reporting)

## Troubleshooting

### Common Issues

1. **Azure Authentication Errors**: Verify your endpoint URLs and API keys
2. **Neo4j Connection Issues**: Ensure Neo4j is running and credentials are correct
3. **Embedding Dimension Mismatch**: Ensure consistent embedding model usage

### Vector Search Performance

For better performance with large datasets:
- HNSW is enabled automatically when embedding dimensions are <= 2000
- Use `text-embedding-3-small` (1536 dims) to allow HNSW in pgvector
- GIN index on `to_tsvector('english', text)` speeds up keyword search
- Adjust `top_k` and similarity thresholds

### Keyword Search Details

The system uses PostgreSQL's full-text search with **flexible OR-based matching**:
- Query words are automatically joined with OR operators (`word1 | word2 | word3`)
- Chunks matching **any** query word are included (not requiring all words)
- Results are ranked by relevance: chunks with more matching words score higher
- Uses `ts_rank_cd` for ranking and `to_tsquery` for flexible matching


