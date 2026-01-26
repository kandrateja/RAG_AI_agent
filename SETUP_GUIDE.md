# Complete Setup and Testing Guide

This guide walks you through setting up and testing the RAG AI Agent system from scratch.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.9+** installed
- [ ] **Docker** (for Postgres + Neo4j containers)
- [ ] **Azure Account** with:
  - Azure Document Intelligence resource
  - Azure OpenAI resource with deployments for:
    - Chat completion model (e.g., GPT-4o, vision-capable)
    - Embedding model (`text-embedding-3-small`)
- [ ] **Surf API** credentials (optional, for web search fallback)
- [ ] At least one **scanned PDF document** to test with

---

## Step 1: Database Setup

### 1.1 Setup PostgreSQL with pgvector (Docker)

```bash
# Pull PostgreSQL image with pgvector
docker pull pgvector/pgvector:pg16

# Run PostgreSQL container
docker run -d \
  --name postgres-rag \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Verify it's running
docker ps | grep postgres-rag
```

### 1.2 Setup Neo4j (Docker)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-strong-password \
  -v neo4j_data:/data \
  neo4j:latest

# Access Neo4j Browser at http://localhost:7474
```

**Note:** Neo4j does not allow the default password `neo4j`. Use a strong password.

---

## Step 2: Python Environment Setup

### 2.1 Navigate to Project Directory

```bash
cd /Users/tejakandra/Downloads/AI-project-app/RAG_AI_agent
```

### 2.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### 2.3 Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Expected output:** All packages should install successfully. This may take a few minutes.

**Verify installation:**

```bash
python -c "import psycopg; import neo4j; import openai; print('All imports successful!')"
```

---

## Step 3: Environment Configuration

### 3.1 Create `.env` File

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file
# On macOS/Linux:
nano .env
# or
code .env  # if you have VS Code

# On Windows:
notepad .env
```

### 3.2 Fill in Your Credentials

Edit `.env` with your actual values:

```env
# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-actual-key-here

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-actual-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # vision-capable deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
NEO4J_DATABASE=neo4j

# Postgres (pgvector)
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag
# Adjust if your Postgres credentials are different

# Application Settings (optional - defaults are fine)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4096
TEMPERATURE=0.7

# Web Search (Surf-like API) - OPTIONAL
# Leave these empty if you don't have Surf API
SURF_API_ENDPOINT=
SURF_API_KEY=
SURF_MAX_RESULTS=5
```

**Important:** 
- Replace all `your-*` placeholders with actual values
- Ensure Postgres DSN matches your database credentials
- Ensure Neo4j password matches your Neo4j instance

### 3.3 Verify Configuration

```bash
# Test that config loads correctly
python -c "from config import settings; print('Config loaded successfully!')"
```

---

## Step 4: Database Initialization

The system will automatically create tables/indexes on first run, but you can verify:

### 4.1 Verify PostgreSQL Connection

```bash
# Test Postgres connection
psql -d rag -c "SELECT version();"
psql -d rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### 4.2 Verify Neo4j Connection

```bash
# Test Neo4j connection (using cypher-shell if installed)
cypher-shell -a bolt://localhost:7687 -u neo4j -p your-password "RETURN 1;"
```

Or test via Python:

```bash
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'your-password'))
with driver.session() as session:
    result = session.run('RETURN 1 as test')
    print('Neo4j connection successful!', result.single()['test'])
driver.close()
"
```

---

## Step 5: Start the API + UI (Recommended)

```bash
# From project root directory
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Open the UI at `http://localhost:8000/`.

### 5.1 Ingest via UI (Recommended)

1. Open the **Ingestion** panel in the left sidebar.
2. Upload a scanned PDF.
3. (Optional) Set a `doc_id`.
4. Click **Ingest** and wait for the success message.

**Note on deduplication:** If you re-ingest the same PDF, it may be skipped because the document hash is stored in Postgres. To re-ingest, clear the `documents` and `chunks` tables in Postgres.

**Note on vision captions:** If you enabled page-level image captions, you must re-ingest to store the merged page text. The system renders each PDF page as an image (PyMuPDF) and captions it with a vision-capable model (e.g., `gpt-4o`).

### 5.2 Verify Data in Databases (Optional)

**Check Postgres:**

```bash
psql -d rag -c "SELECT COUNT(*) as total_chunks FROM chunks;"
psql -d rag -c "SELECT doc_id, COUNT(*) as chunks FROM chunks GROUP BY doc_id;"
```

**Check Neo4j:**

Open Neo4j Browser (http://localhost:7474) and run:

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

## Step 6: Query via UI (Recommended)

1. In the chat box, ask a question about your documents.
2. Review the answer and citations shown under the response.
3. If the question is outside your internal docs, the system will use web search (if configured).

---

## Step 7: CLI (Optional)

If you prefer a script-based flow, use the included CLI test files:

```bash
python test_ingest.py
python test_query.py
```

## Step 8: HTTP API (Optional)

### 8.1 Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","agent_initialized":true}
```

### 8.2 Ingest Document via API

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@test_documents/test.pdf" \
  -F "doc_id=test-doc-001"
```

### 8.3 Query via API

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main topics in the documents?",
    "top_k": 5,
    "use_graph_context": true
  }'
```

### 8.4 Access API Documentation

Open in browser:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Step 9: Troubleshooting

### Common Issues

#### Issue: "Failed to connect to Neo4j"

**Solution:**
```bash
# Check Neo4j is running
docker ps | grep neo4j
# or check Neo4j Desktop

# Verify credentials in .env
# Test connection manually
```

#### Issue: "psycopg.OperationalError: connection refused"

**Solution:**
```bash
# Check Postgres is running
docker ps | grep postgres
# or
pg_isready

# Verify POSTGRES_DSN in .env matches your setup
```



## Summary Checklist

- [ ] PostgreSQL (Docker) running
- [ ] Neo4j (Docker) running
- [ ] Python virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with all credentials
- [ ] Database connections verified
- [ ] API server running and accessible
- [ ] UI ingestion tested
- [ ] UI chat query tested

---

## Quick Reference Commands

```bash
# Start services
docker start postgres-rag neo4j

# Activate virtual environment
source venv/bin/activate

# Start API server
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# Open UI
open http://localhost:8000/

# Optional CLI
python test_ingest.py
python test_query.py

# Check Postgres data
psql -d rag -c "SELECT COUNT(*) FROM chunks;"

# Access Neo4j Browser
open http://localhost:7474
```

---


