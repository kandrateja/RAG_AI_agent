# RAG AI Agent - Hybrid LLM Knowledge Agent

A comprehensive Retrieval-Augmented Generation (RAG) system that ingests scanned PDF documents (including handwritten forms) into a hybrid knowledge base and provides a chat-based interface for question answering.

## 🎯 Key Features

- **Document Ingestion**: Supports scanned PDFs, handwritten forms, tables, Arabic text, and mixed content
- **Hybrid Knowledge Base**: Vector search (PostgreSQL + pgvector) + Graph database (Neo4j)
- **Smart Retrieval**: Prioritizes internal knowledge, falls back to web search when needed
- **Complete Provenance**: All answers include citations with document ID, page number, and chunk ID
- **Form Extraction**: Extracts checkbox states, table data, and handwritten text with high accuracy

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG AI Agent System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Docling   │───▶│   Vision    │───▶│   Titan     │───▶│  PostgreSQL │  │
│  │  (Tables,   │    │  (Claude)   │    │ Embeddings  │    │  + pgvector │  │
│  │   OCR)      │    │ Handwritten │    │  (1024-d)   │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                                                        │          │
│         │                                                        │          │
│         ▼                                                        ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Entity    │───▶│   Neo4j     │◀──▶│   RAG       │◀──▶│   Claude    │  │
│  │  Extractor  │    │   Graph     │    │   Agent     │    │   (LLM)     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                              │                              │
│                                              ▼                              │
│                                       ┌─────────────┐                       │
│                                       │   SERPAPI   │                       │
│                                       │ (Web Search)│                       │
│                                       └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

**See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete step-by-step instructions.**

### Prerequisites

- Docker and Docker Compose
- AWS Account with Bedrock access (Claude Sonnet 4, Titan Embeddings)
- SERPAPI account for web search

### Quick Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd RAG_AI_agent

# 2. Configure environment
cp .env.example .env
# Edit .env with your AWS and SERPAPI credentials

# 3. Start databases
docker-compose up -d postgres-rag neo4j-rag

# 4. Install dependencies & run
pip install -r requirements.txt
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# 5. Access UI
open http://localhost:8000/
```

## 📦 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | AWS Bedrock Claude Sonnet 4 | Text generation, vision, entity extraction |
| **Embeddings** | AWS Titan Multimodal | 1024-d embeddings for text and images |
| **Vector DB** | PostgreSQL + pgvector | Semantic similarity search with HNSW index |
| **Graph DB** | Neo4j | Entity-relationship storage and graph queries |
| **Document Processing** | Docling | PDF extraction, tables, OCR |
| **OCR** | Tesseract | Arabic + English text extraction |
| **Web Search** | SERPAPI | External knowledge retrieval |
| **API** | FastAPI | REST API and web interface |

## 📁 Project Structure

```
RAG_AI_agent/
├── src/
│   ├── api/
│   │   └── server.py                 # FastAPI server with UI
│   ├── ingestion/
│   │   └── docling_processor.py      # Document extraction (PDF, tables, OCR)
│   ├── embeddings/
│   │   └── titan_multimodal.py       # AWS Titan embeddings
│   ├── llm/
│   │   └── bedrock_client.py         # AWS Bedrock Claude client
│   ├── graphrag/
│   │   ├── neo4j_client.py           # Neo4j graph operations
│   │   ├── text_chunker.py           # Text chunking with overlap
│   │   └── graph_confidence.py       # Graph-based confidence scoring
│   ├── vectorstore/
│   │   └── postgres_pgvector.py      # pgvector operations
│   ├── websearch/
│   │   └── surf_client.py            # SERPAPI web search
│   ├── ner/
│   │   └── entity_extractor.py       # LLM-based entity extraction
│   └── rag_agent.py                  # Main orchestrator
├── config.py                          # Configuration from .env
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Database containers
├── .env.example                       # Environment template
├── SETUP_GUIDE.md                     # Detailed setup instructions
└── README.md                          # This file
```

## ✅ Functional Requirements Compliance

### 1. Document Ingestion Pipeline ✅
- **OCR extraction**: Docling + Tesseract (Arabic/English) + Vision fallback
- **Text preprocessing**: Chunking with page number preservation
- **Embedding generation**: AWS Titan Multimodal (1024 dimensions)
- **Vector storage**: PostgreSQL + pgvector with HNSW index
- **Entity extraction**: LLM-based extraction of persons, organizations, locations, concepts
- **Graph storage**: Neo4j with entity-relationship links
- **Chunk metadata**: doc_id, page_number, chunk_id, text content
- **Deduplication**: Hash-based prevention of duplicate ingestion

### 2. Hybrid Knowledge Base ✅
- **Vector index**: pgvector with hybrid semantic + keyword search
- **Graph database**: Neo4j with entities and relationships
- **Graph used at query time**: ChunkRef → Entity expansion for context enrichment

### 3. Query & Agent Interface ✅
- **Chat interface**: FastAPI web UI at `http://localhost:8000/`
- **Internal KB first**: Vector search prioritized
- **Tool calling**: KB search, graph queries, web search
- **Fallback logic**: Web search only when internal knowledge insufficient

### 4. Answer Format & Provenance ✅
- **Clear answers**: Concise responses with context
- **Citations**: Document ID, page number, chunk ID
- **Source indication**: `provenance` field shows internal/online/both/none
- **No hallucination**: Explicit statement when answer not found

### 5. Observability & Reliability ✅
- **Logging**: Comprehensive logging of all retrieval steps
- **Error handling**: OCR failures, empty results, tool failures handled gracefully

## 🔧 Configuration

### Environment Variables

```env
# AWS Bedrock (Claude for LLM + Vision)
USE_BEDROCK_LLM=true
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# AWS Titan Embeddings
USE_TITAN_EMBEDDINGS=true
TITAN_EMBEDDING_MODEL_ID=amazon.titan-embed-image-v1

# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=rag-neo4j-password-2024
NEO4J_DATABASE=neo4j

# PostgreSQL + pgvector
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag

# Web Search (SERPAPI)
SURF_API_ENDPOINT=https://serpapi.com/search.json
SURF_API_KEY=your-serpapi-key
SURF_MAX_RESULTS=5

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI interface |
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest PDF document |
| `/query` | POST | Query the knowledge base |
| `/docs` | GET | OpenAPI documentation |

### Query Request Example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Was an interpreter needed for this case?",
    "top_k": 5,
    "use_graph_context": true
  }'
```

### Query Response Example

```json
{
  "question": "Was an interpreter needed for this case?",
  "answer": "No, an interpreter was not needed. The form shows 'Interpreter needed?: No [TICKED]'",
  "citations": [
    {
      "chunk_id": "doc123_chunk_2",
      "doc_id": "doc123",
      "page_number": 2,
      "similarity": 0.89
    }
  ],
  "provenance": "internal",
  "sources_used": {
    "vector": true,
    "graph": true,
    "web": false
  }
}
```

## 📄 Supported Document Types

| Type | Support Level | Notes |
|------|---------------|-------|
| **Standard PDFs** | ✅ Full | Text extraction via Docling |
| **Scanned PDFs** | ✅ Full | OCR via Tesseract + Vision |
| **Handwritten Forms** | ✅ Full | Vision model transcription |
| **Tables** | ✅ Full | Multi-column extraction preserved |
| **Checkboxes** | ✅ Full | [TICKED]/[EMPTY] state detection |
| **Arabic Text** | ✅ Full | RTL support, cross-lingual retrieval |
| **Mixed Content** | ✅ Full | Printed + handwritten together |

## 🐛 Troubleshooting

### Common Issues

1. **AWS Bedrock errors**: Verify AWS credentials and region
2. **Neo4j connection failed**: Ensure Neo4j container is running
3. **Empty extraction**: Check if PDF is scanned (needs OCR/Vision)
4. **Missing table columns**: Re-ingest with latest code (vision fallback)

### Useful Commands

```bash
# Check container status
docker ps

# View API logs
docker logs -f rag-api

# Clear databases for fresh start
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"
```

## 📚 Additional Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions
- **[.env.example](.env.example)** - Environment variable template

## 📝 License

This project is for assessment purposes.
