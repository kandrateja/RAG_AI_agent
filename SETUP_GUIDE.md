# Setup and Testing Guide

Complete step-by-step guide for setting up and running the RAG AI Agent system.

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Docker** and **Docker Compose** installed
- [ ] **Python 3.9+** installed (only for Option B - local development)
- [ ] **AWS Account** with Bedrock access for:
  - Claude Sonnet 4 (LLM + Vision)
  - Titan Multimodal Embeddings (V1)
  - Titan Text Embeddings (V2) - for multilingual support--optional
- [ ] **SERPAPI Account** for web search (get key from https://serpapi.com/)
- [ ] PDF documents to test with (scanned, handwritten, Arabic, or standard)

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
# Choose V1 only(manual by llm) or V2 for multilingual support(automatic)
# ============================================
USE_TITAN_EMBEDDINGS=true
TITAN_EMBEDDING_MODEL_ID=amazon.titan-embed-image-v1

# V2 for native multilingual (recommended for Arabic documents)
USE_TITAN_V2_FOR_TEXT=true

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
# ARABIC / MULTILINGUAL SETTINGS
# ============================================
# If V2 is disabled, enable these for Arabic support via translation
TRANSLATE_ARABIC_FOR_EMBEDDING=true
USE_ARABIC_SENTENCE_CHUNKING=true

# ============================================
# APPLICATION SETTINGS
# ============================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
INGESTION_PIPELINE_HINT=auto
```

### 2.3 Embedding Model Selection

| Configuration | Use Case | Arabic Support |
|--------------|----------|----------------|
| `USE_TITAN_V2_FOR_TEXT=true` | Multilingual documents | ✅ Native |
| `USE_TITAN_V2_FOR_TEXT=false` + `TRANSLATE_ARABIC_FOR_EMBEDDING=true` | Cross-lingual demonstration | ✅ Via translation |
| `USE_TITAN_V2_FOR_TEXT=false` + `TRANSLATE_ARABIC_FOR_EMBEDDING=false` | English only | ❌ |

**Recommendation**: Use V2 for production Arabic support. Use V1+translation to demonstrate cross-lingual retrieval understanding.

---

## Step 3: Choose Your Setup Method

### Option A: Full Docker (Recommended)

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

### Option B: Local Development

Use this for development with hot-reload.

#### 3B.1 Start databases only

```bash
docker-compose up -d postgres-rag neo4j-rag
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

#### 3B.3 Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3B.4 Install Tesseract OCR (for scanned PDFs)

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng
```

#### 3B.5 Start the application

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
[INIT] Titan Embeddings: V2 for text (multilingual), V1 for images
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 4: Access the System

| URL | Description |
|-----|-------------|
| http://localhost:8000/ | Web UI (main interface) |
| http://localhost:8000/docs | API Documentation (Swagger) |
| http://localhost:8000/health | Health Check |
| http://localhost:7474/ | Neo4j Browser (graph visualization) |

---

## Step 5: Test Document Ingestion

### 5.1 Document Types to Test

| Document Type | What to Check |
|--------------|---------------|
| **Standard PDF** | Text extraction, chunking |
| **Scanned PDF** | OCR extraction (Tesseract) |
| **Handwritten Form** | Vision fallback, checkbox detection |
| **Arabic Document** | Arabic text extraction, V1/V2 embedding |
| **Text + Images PDF** | Figure extraction, multimodal embedding |

### 5.2 Upload a PDF

1. Open http://localhost:8000/
2. Click the **Upload** area or drag a PDF file
3. Click **Ingest Document**

### 5.3 Monitor Ingestion Logs

Watch the terminal for:

```
[INGEST] Processing document: example.pdf
[INGEST] Extracted 10 pages of text
[INGEST] Document detected as predominantly Arabic
[INGEST] Arabic document → Titan V2 (native multilingual)  # or V1 with translation
[INGEST] Created 25 chunks
[INGEST] Stored in PostgreSQL and Neo4j
```

### 5.4 Verify Ingestion

```bash
# Check chunk count
docker exec postgres-rag psql -U postgres -d rag -c "SELECT COUNT(*) FROM chunks;"

# View sample chunks
docker exec postgres-rag psql -U postgres -d rag -c "SELECT chunk_id, page_number, LEFT(text, 100) FROM chunks LIMIT 5;"
```

---

## Step 6: Test Queries

### 6.1 Standard Document Questions

| Question | Expected Behavior |
|----------|------------------|
| Question about ingested content | Returns answer with citations |
| Question not in documents | Triggers web search fallback |
| "What is the weather today?" | Web search only |

### 6.2 Arabic Document Questions

If you've ingested an Arabic document:

**English Questions (Cross-lingual):**
- "What is the role of the International Court of Justice?"
- "What cases are mentioned in the document?"

**Arabic Questions:**
- "ما هو دور محكمة العدل الدولية؟"
- "ما هي القضايا المذكورة في الوثيقة؟"

### 6.3 Check Response Quality

Look for:
- `provenance: "internal"` - Answer from ingested documents
- `citations` - List of sources with page numbers
- `decision_trace.is_cross_lingual` - True for Arabic documents
- `decision_trace.effective_high_threshold` - 0.4 for cross-lingual, 0.7 for same-language

---

## Step 7: Switching Between V1 and V2

### Switch to V2 (Native Multilingual)

1. Edit `.env`:
   ```env
   USE_TITAN_V2_FOR_TEXT=true
   ```

2. Clear database (embeddings are incompatible):
   ```bash
   docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
   docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"
   ```

3. Restart server

4. Re-ingest documents

### Switch to V1 (Translation-based)

1. Edit `.env`:
   ```env
   USE_TITAN_V2_FOR_TEXT=false
   TRANSLATE_ARABIC_FOR_EMBEDDING=true
   USE_ARABIC_SENTENCE_CHUNKING=true
   ```

2. Clear database and restart

3. Re-ingest documents

---

## Step 8: Verify Data in Databases

### 8.1 Check PostgreSQL

```bash
docker exec -it postgres-rag psql -U postgres -d rag

# Count chunks
SELECT COUNT(*) FROM chunks;

# View chunks by document
SELECT doc_id, COUNT(*) as chunks FROM chunks GROUP BY doc_id;

# Check embedding dimensions
SELECT chunk_id, array_length(embedding, 1) as dim FROM chunks LIMIT 1;

# Exit
\q
```

### 8.2 Check Neo4j

1. Open http://localhost:7474/
2. Login: username `neo4j`, password `rag-neo4j-password-2024`
3. Run queries:

```cypher
// Count entities
MATCH (e) WHERE 'Entity' IN labels(e) RETURN count(e);

// View entities by type
MATCH (e) WHERE 'Entity' IN labels(e) RETURN e.type, count(e) ORDER BY count(e) DESC;

// View relationships
MATCH (e1)-[r]->(e2) 
WHERE 'Entity' IN labels(e1) AND 'Entity' IN labels(e2)
RETURN e1.name, type(r), e2.name LIMIT 10;
```

---

## Step 9: Stop the System

### Stop everything

```bash
docker-compose down
```

### Stop and remove all data

```bash
docker-compose down -v
```

### Clear data only (keep containers)

```bash
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"
```

---

## Troubleshooting

### Issue: AWS Bedrock Access Denied

**Solution:**
1. Verify AWS credentials in `.env`
2. Ensure Bedrock model access is enabled in AWS Console
3. Check IAM role has `bedrock:InvokeModel` permission

### Issue: Low Similarity Scores for Arabic

**Symptoms:** Scores around 0.3-0.4 for Arabic documents

**Solution:**
- The system automatically adjusts thresholds for cross-lingual retrieval
- Check logs for: `[QUERY] Cross-lingual retrieval detected - using threshold 0.4`
- Consider switching to V2 for native multilingual support

### Issue: "Could not find sufficient information"

**Solution:**
1. Check that documents were ingested successfully
2. Verify the question relates to ingested content
3. Check logs for threshold values and similarity scores
4. For Arabic docs, ensure cross-lingual detection is working

### Issue: Empty Extraction from Scanned PDF

**Solution:**
1. Ensure Tesseract OCR is installed
2. Check logs for OCR messages
3. Vision fallback should trigger automatically

### Issue: V1/V2 Embedding Mismatch

**Symptoms:** "Could not find information" after changing V1/V2 setting

**Solution:**
- V1 and V2 embeddings are in different vector spaces
- Must clear database and re-ingest after switching models

---

## Quick Reference Commands

### Full Docker

```bash
docker-compose up -d          # Start all
docker-compose ps             # Check status
docker-compose logs -f        # View logs
docker-compose down           # Stop
docker-compose down -v        # Stop + remove data
```

### Local Development

```bash
docker-compose up -d postgres-rag neo4j-rag   # Start DBs
source venv/bin/activate                        # Activate venv
uvicorn src.api.server:app --reload --port 8000 # Run server
```

### Database Operations

```bash
# Clear all data
docker exec postgres-rag psql -U postgres -d rag -c "DELETE FROM chunks; DELETE FROM documents;"
docker exec neo4j-rag cypher-shell -u neo4j -p 'rag-neo4j-password-2024' "MATCH (n) DETACH DELETE n;"

# Check counts
docker exec postgres-rag psql -U postgres -d rag -c "SELECT COUNT(*) FROM chunks;"
```

---

## Demo Checklist

For your assignment demo, show:

### 1. Document Ingestion
- [ ] Upload a PDF (scanned/handwritten/Arabic/standard)
- [ ] Show logs with extraction details
- [ ] Verify data in PostgreSQL and Neo4j

### 2. Query Internal Knowledge
- [ ] Ask questions about ingested document
- [ ] Show citations (doc_id, page_number, chunk_id)
- [ ] Show `provenance: "internal"`

### 3. Cross-Lingual Retrieval (if using Arabic)
- [ ] Ingest Arabic document
- [ ] Query in English
- [ ] Show Arabic text in citations
- [ ] Show `is_cross_lingual: true` in decision trace

### 4. Web Search Fallback
- [ ] Ask question not in documents
- [ ] Show web search triggered
- [ ] Show `provenance: "online"`

### 5. Special Features
- [ ] Checkbox extraction (`[TICKED]`/`[EMPTY]`)
- [ ] Table extraction
- [ ] Figure-only embedding (for text+image PDFs)
- [ ] Handwritten text transcription
