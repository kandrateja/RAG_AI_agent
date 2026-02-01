# RAG AI Agent - Hybrid LLM Knowledge Agent

A comprehensive Retrieval-Augmented Generation (RAG) system that ingests documents (including scanned PDFs, handwritten forms, and Arabic text) into a hybrid knowledge base and provides a chat-based interface for question answering.

## 🎯 Key Features

- **Multi-Document Support**: Standard PDFs, scanned documents, handwritten forms, tables, and Arabic text
- **Hybrid Knowledge Base**: Vector search (PostgreSQL + pgvector) + Graph database (Neo4j)
- **Multilingual Embeddings**: Dual embedding model support (V1 multimodal + V2 multilingual)
- **Figure Extraction**: Extracts and embeds only figures/diagrams from text+image PDFs (avoids duplication)
- **Cross-lingual Retrieval**: Query Arabic documents in English (and vice versa)
- **Smart Retrieval**: Prioritizes internal knowledge, falls back to web search when needed
- **Complete Provenance**: All answers include citations with document ID, page number, and chunk ID

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAG AI Agent System                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Document Ingestion Pipeline                       │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │  PDF → Docling → OCR (Tesseract) → Vision (Claude) → Text Extraction   │    │
│  │         ↓              ↓                ↓                               │    │
│  │  [Tables]      [Arabic/English]   [Handwritten]                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Embedding & Storage                               │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  Text Chunks ──→ Titan V1/V2 ──→ PostgreSQL + pgvector                  │    │
│  │       │              │                    │                              │    │
│  │       │         (V1: English,        (1024-d vectors,                   │    │
│  │       │          V2: Multilingual)    HNSW index)                       │    │
│  │       │                                                                  │    │
│  │  Figures Only ──→ Titan V1 ──→ (Multimodal embeddings)                  │    │
│  │                                                                          │    │
│  │  Entities/Relationships ──→ Neo4j Graph Database                        │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Query Processing                                  │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                          │    │
│  │  User Query ──→ Hybrid Search ──→ Graph Expansion ──→ Claude LLM        │    │
│  │       │         (Semantic +           │                    │            │    │
│  │       │          Keyword)             │                    │            │    │
│  │       │              │                │                    │            │    │
│  │       └──────────────┴────────────────┴────────────────────┘            │    │
│  │                              │                                           │    │
│  │                    (If insufficient) → SERPAPI Web Search               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

**See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete step-by-step instructions.**

### Prerequisites

- Docker and Docker Compose
- AWS Account with Bedrock access (Claude Sonnet 4, Titan Embeddings V1 & V2)
- SERPAPI account for web search

### Quick Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd RAG_AI_agent

# 2. Configure environment
cp .env.example .env
# Edit .env with your AWS and SERPAPI credentials

# 3. Start all services (databases + API)
docker-compose up -d

# 4. Access UI
open http://localhost:8000/
```

### Alternative: Local Development

```bash
# Start databases only
docker-compose up -d postgres-rag neo4j-rag

# Install dependencies
pip install -r requirements.txt

# Run server with hot-reload
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

## 📦 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | AWS Bedrock Claude Sonnet 4 | Text generation, vision, entity extraction |
| **Text Embeddings (V1)** | AWS Titan Multimodal (image-v1) | 1024-d embeddings for English text + images |
| **Text Embeddings (V2)** | AWS Titan Text (text-v2) | 1024-d embeddings for multilingual text (100+ languages) |
| **Vector DB** | PostgreSQL + pgvector | Semantic + keyword hybrid search with HNSW index |
| **Graph DB** | Neo4j | Entity-relationship storage and graph queries |
| **Document Processing** | Docling | PDF extraction, tables, layout analysis |
| **OCR** | Tesseract | Arabic + English text extraction from scanned documents |
| **Web Search** | SERPAPI | External knowledge retrieval |
| **API** | FastAPI | REST API and web interface |

## 📁 Project Structure

```
RAG_AI_agent/
├── src/
│   ├── api/
│   │   └── server.py                 # FastAPI server with UI
│   ├── ingestion/
│   │   └── docling_processor.py      # Document extraction (PDF, tables, OCR, figures)
│   ├── embeddings/
│   │   └── titan_multimodal.py       # AWS Titan embeddings (V1 + V2 support)
│   ├── llm/
│   │   └── bedrock_client.py         # AWS Bedrock Claude client
│   ├── graphrag/
│   │   ├── neo4j_client.py           # Neo4j graph operations
│   │   ├── text_chunker.py           # Text chunking (standard + Arabic sentence-aware)
│   │   └── graph_confidence.py       # Graph-based confidence scoring
│   ├── vectorstore/
│   │   └── postgres_pgvector.py      # pgvector operations with hybrid search
│   ├── websearch/
│   │   └── surf_client.py            # SERPAPI web search
│   ├── ner/
│   │   └── entity_extractor.py       # LLM-based entity extraction
│   └── rag_agent.py                  # Main orchestrator
├── ui/
│   ├── index.html                    # Web interface
│   ├── app.js                        # Frontend logic
│   └── styles.css                    # Styling
├── config.py                         # Configuration from .env
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container build
├── docker-compose.yml                # Full stack deployment
├── docker-compose.dev.yml            # Databases only (for local dev)
├── .env.example                      # Environment template
├── SETUP_GUIDE.md                    # Detailed setup instructions
└── README.md                         # This file
```

## 🌐 Multilingual Support

### Embedding Model Options

| Feature | V1 (titan-embed-image-v1) | V2 (titan-embed-text-v2) |
|---------|---------------------------|--------------------------|
| **Text Embedding** | ✅ English only | ✅ 100+ languages |
| **Image Embedding** | ✅ Yes (multimodal) | ❌ No |
| **Arabic Support** | Via translation | ✅ Native |
| **Dimensions** | 1024 | 1024 |

### Configuration

```env
# V1 Only (with Arabic translation)
USE_TITAN_V2_FOR_TEXT=false
TRANSLATE_ARABIC_FOR_EMBEDDING=true

# V2 for Multilingual (recommended for Arabic)
USE_TITAN_V2_FOR_TEXT=true
```

### Cross-Lingual Retrieval

When using V1 with translation:
1. Arabic text is translated to English during ingestion
2. Original Arabic text is preserved in the database
3. English queries can retrieve Arabic documents
4. Responses can be in Arabic or English based on the query language

## 📄 Supported Document Types

| Type | Support | Processing Pipeline |
|------|---------|-------------------|
| **Standard PDFs** | ✅ Full | Docling text extraction |
| **Scanned PDFs** | ✅ Full | Tesseract OCR + Vision fallback |
| **Handwritten Forms** | ✅ Full | Claude Vision transcription |
| **Tables** | ✅ Full | Docling table detection + Vision fallback |
| **Checkboxes** | ✅ Full | [TICKED]/[EMPTY] state detection |
| **Arabic Text** | ✅ Full | OCR + V2 native or V1 + translation |
| **Text + Images** | ✅ Full | Text via Docling, figures via multimodal embedding |

## ✅ Functional Requirements Compliance

### 1. Document Ingestion Pipeline ✅
- **OCR extraction**: Docling + Tesseract (Arabic/English) + Vision fallback
- **Text preprocessing**: Smart chunking with page number preservation
- **Arabic chunking**: Sentence-boundary aware chunking for Arabic text
- **Embedding generation**: AWS Titan V1 (multimodal) and/or V2 (multilingual)
- **Figure extraction**: Extracts only figures/diagrams, not full pages (avoids text duplication)
- **Vector storage**: PostgreSQL + pgvector with HNSW index + keyword (BM25) search
- **Entity extraction**: LLM-based extraction of persons, organizations, locations, concepts
- **Graph storage**: Neo4j with entity-relationship links to chunks
- **Deduplication**: Hash-based prevention of duplicate ingestion

### 2. Hybrid Knowledge Base ✅
- **Vector index**: pgvector with hybrid semantic + keyword search
- **Graph database**: Neo4j with entities and relationships
- **Graph used at query time**: ChunkRef → Entity expansion for context enrichment
- **Cross-lingual support**: Arabic documents retrievable via English queries

### 3. Query & Agent Interface ✅
- **Chat interface**: FastAPI web UI at `http://localhost:8000/`
- **Internal KB first**: Vector search prioritized with adaptive thresholds
- **Tool calling**: KB search, graph queries, web search
- **Fallback logic**: Web search only when internal knowledge insufficient
- **Language detection**: Automatic threshold adjustment for cross-lingual queries

### 4. Answer Format & Provenance ✅
- **Clear answers**: Markdown-formatted responses with structure
- **Citations**: Document ID, page number, chunk ID, figure index (for images)
- **Source indication**: `provenance` field shows internal/online/both/none
- **Score type**: "semantic + keyword" for text, "visual similarity" for images
- **No hallucination**: Explicit statement when answer not found

### 5. Observability & Reliability ✅
- **Logging**: Comprehensive logging of all retrieval steps
- **Error handling**: OCR failures, empty results, tool failures handled gracefully
- **Cross-lingual detection**: Logs when Arabic documents detected and thresholds adjusted

## 🔧 Configuration

### Environment Variables

See [.env.example](.env.example) for all options. Key settings:

```env
# AWS Bedrock (Claude for LLM + Vision)
USE_BEDROCK_LLM=true
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# Titan Embeddings
USE_TITAN_EMBEDDINGS=true
USE_TITAN_V2_FOR_TEXT=true  # true for native multilingual, false for V1+translation

# Databases
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=rag-neo4j-password-2024

# Web Search
SURF_API_KEY=your-serpapi-key

# Arabic Settings (only when V2 disabled)
TRANSLATE_ARABIC_FOR_EMBEDDING=true
USE_ARABIC_SENTENCE_CHUNKING=true
```

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI interface |
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest PDF document |
| `/query` | POST | Query the knowledge base |
| `/docs` | GET | OpenAPI documentation |

### Query Response Example

```json
{
  "question": "What is the role of the International Court of Justice?",
  "answer": "The International Court of Justice is the principal judicial organ of the United Nations...",
  "citations": [
    {
      "chunk_id": "doc123_chunk_64",
      "doc_id": "doc123",
      "doc_name": "icj_report.pdf",
      "page_number": 16,
      "similarity": 0.753,
      "score_type": "semantic + keyword"
    }
  ],
  "provenance": "internal",
  "sources_used": {
    "vector": true,
    "graph": false,
    "web": false
  },
  "decision_trace": {
    "is_cross_lingual": true,
    "effective_high_threshold": 0.4,
    "vector_best_score": 0.753
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **AWS Bedrock errors**: Verify AWS credentials and Bedrock model access
2. **Neo4j connection failed**: Ensure Neo4j container is running and healthy
3. **Empty extraction**: Check if PDF is scanned (needs OCR/Vision)
4. **Low similarity scores for Arabic**: Switch to V2 embeddings or ensure translation is enabled
5. **"Could not find sufficient information"**: Check logs for threshold values

### Useful Commands

```bash
# Check container status
docker-compose ps

# View API logs
docker-compose logs -f rag-api

# Clear databases for fresh start
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"

# Check chunk count
docker exec postgres-rag psql -U postgres -d rag -c "SELECT COUNT(*) FROM chunks;"
```

## 📚 Additional Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup and testing instructions
- **[.env.example](.env.example)** - Environment variable reference

## 📝 License

This project is for assessment purposes.
