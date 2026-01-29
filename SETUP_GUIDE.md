# Setup and Testing Guide

Complete step-by-step guide for setting up and running the RAG AI Agent system.

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Docker** and **Docker Compose** installed
- [ ] **Python 3.9+** installed (only for Option B - local development)
- [ ] **AWS Account** with Bedrock access for:
  - Claude Sonnet 4 (LLM + Vision)
  - Titan Multimodal Embeddings
- [ ] **SERPAPI Account** for web search (get key from https://serpapi.com/)
- [ ] At least one **scanned PDF document** to test with

---

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd RAG_AI_agent
```

---

## Step 2: Configure Environment Variables

### 2.1 Create `.env` file

```bash
cp .env.example .env
```

### 2.2 Edit `.env` file with your credentials

```env
# ============================================
# AWS BEDROCK CONFIGURATION (Required)
# ============================================
USE_BEDROCK_LLM=true
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key

# ============================================
# AWS TITAN EMBEDDINGS (Required)
# ============================================
USE_TITAN_EMBEDDINGS=true
TITAN_EMBEDDING_MODEL_ID=amazon.titan-embed-image-v1

# ============================================
# NEO4J GRAPH DATABASE (Required)
# ============================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=rag-neo4j-password-2024
NEO4J_DATABASE=neo4j

# ============================================
# POSTGRESQL + PGVECTOR (Required)
# ============================================
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/rag

# ============================================
# WEB SEARCH - SERPAPI (Required)
# ============================================
SURF_API_ENDPOINT=https://serpapi.com/search.json
SURF_API_KEY=your-serpapi-key
SURF_MAX_RESULTS=5

# ============================================
# APPLICATION SETTINGS (Optional - defaults shown)
# ============================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4096
TEMPERATURE=0.7
INGESTION_PIPELINE_HINT=auto
INGESTION_VISION_FALLBACK_MIN_CHARS=50
TRANSLATE_ARABIC_FOR_EMBEDDING=true
```

### 2.3 Important Notes

- **AWS Credentials**: Must have Bedrock access enabled for Claude and Titan models
- **Neo4j Password**: Must match `rag-neo4j-password-2024` (set in docker-compose.yml)
- **SERPAPI Key**: Get from https://serpapi.com/ (free tier available)

---

## Step 3: Choose Your Setup Method

### Option A: Full Docker (Recommended for Reviewers)

This runs everything in Docker - no Python installation needed.

```bash
# Start the entire stack
docker-compose up -d

# Wait for services to be healthy (1-2 minutes)
docker-compose ps

# View logs
docker-compose logs -f
```

All services should show "Up (healthy)":
```
NAME          STATUS                   PORTS
postgres-rag  Up (healthy)             0.0.0.0:5432->5432/tcp
neo4j-rag     Up (healthy)             0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
rag-api       Up (healthy)             0.0.0.0:8000->8000/tcp
```

**That's it!** Skip to Step 6 to access the system.

---

### Option B: Local Development (Databases in Docker, App Local)

Use this for development with hot-reload.

#### 3B.1 Start databases only

```bash
docker-compose -f docker-compose.dev.yml up -d
```

#### 3B.2 Verify databases are running

```bash
docker ps
```

You should see:
```
CONTAINER ID   IMAGE                    STATUS          PORTS
xxxx           pgvector/pgvector:pg16   Up (healthy)    0.0.0.0:5432->5432/tcp
xxxx           neo4j:5                  Up (healthy)    0.0.0.0:7474->7474/tcp, 0.0.0.0:7687->7687/tcp
```

#### 3B.3 Wait for Neo4j to be ready (first time only)

```bash
docker logs -f neo4j-rag
```

Press `Ctrl+C` once you see "Started."

---

## Step 4: Install Python Dependencies (Option B Only)

Skip this step if you're using Option A (Full Docker).

### 4.1 Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Install Tesseract OCR (for scanned PDFs with Arabic)

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng
```

---

## Step 5: Start the Application (Option B Only)

Skip this step if you're using Option A (Full Docker).

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started reloader process
```

---

## Step 6: Access the System

| URL | Description |
|-----|-------------|
| http://localhost:8000/ | Web UI (main interface) |
| http://localhost:8000/docs | API Documentation (Swagger) |
| http://localhost:8000/health | Health Check |
| http://localhost:7474/ | Neo4j Browser (graph visualization) |

---

## Step 7: Test Document Ingestion

### 7.1 Open Web UI

Navigate to http://localhost:8000/

### 7.2 Upload a PDF

1. Click the **Upload** area or drag a PDF file
2. Wait for the file to upload
3. Click **Ingest Document**

### 7.3 Monitor Ingestion Logs

In the terminal running the server, you'll see:
```
[INGEST] Processing document: example.pdf
[INGEST] Extracted 10 pages of text
[INGEST] Page 1: Form extraction stats:
  - Checkboxes: 5 ticked, 3 empty
  - Table columns (| chars): 48
[INGEST] Extracted 15 entities and 8 relationships
[INGEST] Created 25 chunks
[INGEST] Stored in PostgreSQL and Neo4j
```

### 7.4 Verify Ingestion Success

The response will show:
```json
{
  "doc_id": "example-uuid",
  "status": "success",
  "chunks_created": 25,
  "entities_extracted": 15,
  "relationships_extracted": 8
}
```

---

## Step 8: Test Queries

### 8.1 Sample Test Questions

Based on a handwritten safeguarding form, test these:

| # | Question | Expected Answer |
|---|----------|-----------------|
| 1 | What is the telephone number for Middlesbrough Adult Access Team? | 01642 065070 |
| 2 | Was an interpreter needed for this case? | No (checkbox was ticked for No) |
| 3 | What type of abuse did the patient disclose? | Physical abuse (hitting and kicking) |
| 4 | What is the patient's name and address? | Peter Jones, 1 The Front, Hartlepool |
| 5 | What type of abuse was suspected according to the checkboxes? | Discriminatory |
| 6 | Does the adult have mental capacity? | Yes |

### 8.2 Test Internal Knowledge

Ask a question about your ingested document. The response should include:
- `provenance: "internal"`
- Citations with doc_id, page_number, chunk_id

### 8.3 Test Web Search Fallback

Ask a question NOT in your documents (e.g., "What is the current weather?"). The response should include:
- `provenance: "online"` or `provenance: "both"`
- Web citations with URLs

---

## Step 9: Verify Data in Databases

### 9.1 Check PostgreSQL

```bash
# Connect to PostgreSQL
docker exec -it postgres-rag psql -U postgres -d rag

# Count chunks
SELECT COUNT(*) FROM chunks;

# View sample chunks
SELECT chunk_id, page_number, LEFT(text, 100) FROM chunks LIMIT 5;

# Exit
\q
```

### 9.2 Check Neo4j

1. Open http://localhost:7474/
2. Login: username `neo4j`, password `rag-neo4j-password-2024`
3. Run queries:

```cypher
// Count entities
MATCH (e) WHERE 'Entity' IN labels(e) RETURN count(e);

// View entities
MATCH (e) WHERE 'Entity' IN labels(e) RETURN e.name, e.type LIMIT 10;

// View relationships
MATCH (e1)-[r]->(e2) 
WHERE 'Entity' IN labels(e1) AND 'Entity' IN labels(e2)
RETURN e1.name, type(r), e2.name LIMIT 10;
```

---

## Step 10: Stop the System

### 10.1 Stop the Python server

Press `Ctrl+C` in the terminal running uvicorn

### 10.2 Stop databases

```bash
docker-compose down
```

### 10.3 Remove all data (fresh start)

```bash
# Stop and remove volumes
docker-compose down -v

# Or manually clear data without removing containers
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"
```

---

## Troubleshooting

### Issue: AWS Bedrock Access Denied

```
Error: AccessDeniedException
```

**Solution:**
1. Verify AWS credentials are correct in `.env`
2. Ensure Bedrock model access is enabled in AWS Console
3. Check your IAM role has `bedrock:InvokeModel` permission

### Issue: Neo4j Connection Failed

```
Error: Failed to connect to Neo4j
```

**Solution:**
1. Verify Neo4j container is running: `docker ps`
2. Check password matches: `rag-neo4j-password-2024`
3. Wait for Neo4j to fully start (can take 30-60 seconds)

### Issue: Empty Document Extraction

**Solution:**
1. If PDF is scanned, ensure Tesseract is installed
2. Check terminal logs for vision fallback messages
3. Verify the PDF contains actual content (not blank)

### Issue: Missing Table Columns

**Solution:**
1. Clear database and re-ingest the document
2. Check logs for `[DOCLING TABLE]` messages
3. Vision will automatically extract tables if Docling fails

### Issue: Port Already in Use

```
Error: Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn src.api.server:app --port 8001
```

---

## Quick Reference Commands

### Option A: Full Docker

```bash
# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f rag-api

# Stop everything
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### Option B: Local Development

```bash
# Start databases only
docker-compose -f docker-compose.dev.yml up -d

# Run Python app
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# Stop
Ctrl+C  # Stop uvicorn
docker-compose -f docker-compose.dev.yml down
```

### Database Operations

```bash
# Clear all data (both options)
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"

# Check PostgreSQL
docker exec postgres-rag psql -U postgres -d rag -c "SELECT COUNT(*) FROM chunks;"

# Check Neo4j (open browser)
open http://localhost:7474
```

---

## Summary Checklist

### Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] Repository cloned
- [ ] `.env` file created with:
  - [ ] AWS Bedrock credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
  - [ ] SERPAPI key (SURF_API_KEY)

### Option A: Full Docker
- [ ] Run `docker-compose up -d`
- [ ] All 3 services healthy: `docker-compose ps`
- [ ] Web UI accessible at http://localhost:8000/

### Option B: Local Development
- [ ] Databases started: `docker-compose -f docker-compose.dev.yml up -d`
- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Tesseract OCR installed (for scanned PDFs)
- [ ] Application running: `uvicorn src.api.server:app --reload`
- [ ] Web UI accessible at http://localhost:8000/

### Testing
- [ ] Document ingested successfully
- [ ] Queries returning correct answers with citations
- [ ] Web search fallback working (test with question not in documents)

---

## Demo Recording Checklist

For your assignment demo, show:

1. **Document Ingestion**
   - Upload a scanned/handwritten PDF
   - Show logs with extraction details
   - Verify data in PostgreSQL and Neo4j

2. **Query Internal Knowledge**
   - Ask questions about the ingested document
   - Show citations (doc_id, page_number, chunk_id)
   - Show `provenance: "internal"`

3. **Web Search Fallback**
   - Ask a question not in documents
   - Show web search being triggered
   - Show `provenance: "online"` or `"both"`

4. **Special Features** (optional)
   - Checkbox extraction (`[TICKED]`/`[EMPTY]`)
   - Table extraction with multiple columns
   - Handwritten text transcription
