# Setup and Testing Guide

This guide walks you through setting up and running the RAG AI Agent system using Docker Compose.

## Prerequisites

Before starting, ensure you have:

- [ ] **Docker** and **Docker Compose** installed
- [ ] **Azure Account** with:
  - Azure Document Intelligence resource
  - Azure OpenAI resource with deployments for:
    - Chat completion model (e.g., GPT-4o, vision-capable)
    - Embedding model (`text-embedding-3-small`)
- [ ] **Surf API** credentials (optional, for web search fallback) you get these credentials from here https://serpapi.com/
- [ ] At least one **scanned PDF document** to test with

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd RAG_AI_agent
```

---

## Step 2: Configure Environment Variables

1. **Create `.env` file**:
```bash
cp .env.example .env
```

2. **Edit `.env` file** and fill in your Azure credentials:

```env
# Required: Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-actual-key

# Required: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Database connections (for Docker Compose - use service names)
POSTGRES_DSN=postgresql://postgres:postgres@postgres:5432/rag
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=rag-neo4j-password-2024
NEO4J_DATABASE=neo4j

# Application Settings (optional - defaults are fine)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4096
TEMPERATURE=0.7

# Web Search (Surf-like API) - OPTIONAL
SURF_API_ENDPOINT=
SURF_API_KEY=
SURF_MAX_RESULTS=5
```

**Important Notes:**
- Replace all `your-*` placeholders with actual Azure credentials
- Database connection strings use service names (`postgres`, `neo4j`) not `localhost` for Docker
- **Neo4j password must be**: `rag-neo4j-password-2024` (matches Docker Compose configuration)

---

## Step 3: Start the System

```bash
docker-compose up -d
```

This will:
- Pull required Docker images (first time only)
- Start PostgreSQL with pgvector
- Start Neo4j
- Build and start the RAG API server

**Wait for services to be healthy** (about 30-60 seconds on first run).

Verify services are running:
```bash
docker-compose ps
```

All containers should show as "Up" and "healthy".

---

## Step 4: Access the System

- **Web UI**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Neo4j Browser**: http://localhost:7474/
  - Username: `neo4j`
  - Password: `rag-neo4j-password-2024`



## Step 5: Test the System

### 5.1 Ingest a Document

1. Open the Web UI: http://localhost:8000/
2. Navigate to the **Ingestion** panel in the left sidebar
3. Upload a scanned PDF document
5. Click **Ingest** and wait for the success message

**Note on deduplication:** If you re-ingest the same PDF, it may be skipped because the document hash is stored in Postgres. To re-ingest, clear the `documents` and `chunks` tables in Postgres.

**Note on vision captions:** If you enabled image captions, you must re-ingest to store the merged page text. The system renders each PDF page as an image (PyMuPDF), asks the vision model to detect whether a diagram is present, and only keeps a caption when a diagram/table is detected (then merges it into that page's text).

**Note on hybrid search:** The system uses a hybrid approach combining semantic similarity (70%) and flexible keyword search (30%). **Semantic search** uses cosine similarity with pgvector (HNSW index for fast retrieval). **Keyword search** uses flexible OR-based matching - any word in your question can match chunks (not requiring all words). Results are ranked by relevance - chunks with more matching words score higher. Keyword search uses PostgreSQL's full-text search on the existing `text` column (no separate column needed).

### 5.2 Query the System

1. In the chat box, ask a question about your documents.
   Sample query for pdf_a.pdf: What are muscle spindles, and how do they contribute to proprioception?
2. Review the answer and citations shown under the response.
3. If the question is outside your internal docs, the system will use web search (if configured).

---

## Step 6: Verify Data in Databases (Optional)

### Check Postgres

Connect using TablePlus or any PostgreSQL client:
- **Host**: `localhost`
- **Port**: `5432`
- **Username**: `postgres`
- **Password**: `postgres`
- **Database**: `rag`

Query examples:
```sql
SELECT COUNT(*) as total_chunks FROM chunks;
SELECT doc_id, COUNT(*) as chunks FROM chunks GROUP BY doc_id;
```

### Check Neo4j

1. Open Neo4j Browser: http://localhost:7474/
2. Login with:
   - Username: `neo4j`
   - Password: `rag-neo4j-password-2024`

Run queries:
```cypher
// Count entities
MATCH (e)
WHERE 'Entity' IN labels(e)
RETURN count(e) as entity_count;

// View some entities
MATCH (e)
WHERE 'Entity' IN labels(e)
RETURN e.name, e.type
LIMIT 10;

// View relationships
MATCH (e1)-[r]->(e2)
WHERE 'Entity' IN labels(e1) AND 'Entity' IN labels(e2)
RETURN e1.name, type(r), e2.name
LIMIT 10;
```

---

## Step 7: Stop the System

```bash
docker-compose down
```

To remove all data (fresh start):
```bash
docker-compose down -v
```

---

## Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose logs

# Check specific service logs
docker-compose logs rag-api
docker-compose logs postgres
docker-compose logs neo4j

# Restart services
docker-compose restart
```

### API server errors

- Verify `.env` file has correct Azure credentials
- Check database connections are using service names (`postgres`, `neo4j`)
- Ensure Neo4j password in `.env` matches: `rag-neo4j-password-2024`

### Port conflicts

If ports 5432, 7474, 7687, or 8000 are already in use:
- Stop conflicting services, or
- Update port mappings in `docker-compose.yml`

### Neo4j authentication errors

If you see "The client is unauthorized due to authentication failure":
- Ensure `NEO4J_PASSWORD=rag-neo4j-password-2024` in your `.env` file
- Restart the API container: `docker-compose restart rag-api`

---

## Development Mode (Optional)

To run only databases in Docker and Python app locally:

```bash
# Start only databases
docker-compose -f docker-compose.dev.yml up -d

# Update .env to use localhost for databases
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag
NEO4J_URI=bolt://localhost:7687

# Run Python app locally
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Quick Reference

```bash
# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Remove all data
docker-compose down -v

# Restart a specific service
docker-compose restart rag-api
```

---

## Summary Checklist

- [ ] Docker and Docker Compose installed
- [ ] Repository cloned
- [ ] `.env` file created and configured with Azure credentials
- [ ] Database connection strings updated for Docker (service names)
- [ ] Neo4j password set to `rag-neo4j-password-2024`
- [ ] `docker-compose up -d` executed successfully
- [ ] All services running and healthy
- [ ] Web UI accessible at http://localhost:8000/
- [ ] Document ingested successfully
- [ ] Query tested and working
