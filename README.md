# Deficiency Chatbot - Easy Backend Guide

This guide explains the backend in simple language.

It is written for:
- business users who want to understand "what happens when I ask a question"
- developers who need technical clarity to maintain or integrate the API

Frontend is intentionally kept minimal. This document focuses on backend logic.

## What This System Does

This backend supports **two types of Q&A**:

1. **Structured Data Q&A** (`/chat`)
   - User asks questions about deficiency Excel data.
   - Backend uses OpenAI + tool calling.
   - Tools query the DataFrame and return exact data.

2. **Document Q&A** (`/api/documents/chat`)
   - User uploads PDF/Word.
   - Backend extracts text and stores it in memory.
   - Claude on Vertex answers from document text only.

---

## High-Level Backend Components

- `backend/api.py`
  - Main FastAPI app, request/response models, all endpoints.
- `backend/llm_service.py`
  - OpenAI chat orchestration and tool-call loop for `/chat`.
- `backend/tools.py`
  - Tool definitions and mapping to Python functions.
- `backend/data_service.py`
  - Loads Excel, preprocesses, runs all data queries.
- `backend/feedback_store.py`
  - Logs query history and feedback in SQLite.
- `backend/document_service.py`
  - Extracts text from PDF/DOCX and stores docs in memory.

---

## Step 1: Install and Run Backend

## Prerequisites

- Python 3.9+
- `uv` (recommended) or `pip`
- OpenAI API key (for structured `/chat`)
- GCP service account JSON (for document Q&A on Vertex Claude)

## Install dependencies

From project root:

```bash
uv sync
source .venv/bin/activate
```

Alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure `.env`

Create `.env` in project root.

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Optional: override default data file
# DATA_FILE_PATH=/absolute/path/to/deficiency_data.xlsx

API_HOST=0.0.0.0
API_PORT=8000

# For document Q&A (Claude on Vertex)
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCP_REGION=asia-southeast1
CLAUDE_MODEL=claude-4-6@<snapshot>
```

## Start backend

```bash
source .venv/bin/activate
uvicorn backend.api:app --reload --port 8000
```

Check health:

- Open `http://localhost:8000/health`

If working, you should see healthy status and loaded record count.

---

## Step 2: What Happens Internally When User Asks a Question

This is the most important flow.

## A) Structured Data Q&A (`POST /chat`)

Imagine user asks:

> "How many deficiencies are there for USA?"

Here is the exact backend flow in simple steps:

1. Request hits `POST /chat` in `backend/api.py`.
2. API creates a `query_id` and starts a timer.
3. API forwards message + history to `LLMService.chat()`.
4. `LLMService` sends prompt to OpenAI with available tools.
5. Model decides if it needs tools (usually yes for data questions).
6. If tool call is returned:
   - parse tool name and arguments
   - run `execute_tool()` in `backend/tools.py`
   - tool maps to `DataService` function
   - function queries DataFrame and returns structured result
7. Tool result is sent back to model as `tool` message.
8. Model may call more tools (loop continues up to `max_tool_calls=8`).
9. When model has enough data, it generates final natural-language answer.
10. API logs everything to SQLite (`query_logs`) via `feedback_store`.
11. API returns response payload:
    - `response` (human-readable answer)
    - `tool_calls` (which tools were used)
    - `data` (structured records/counts)
    - `query_id` (for feedback)

### Why this gives reliable answers

- LLM does not guess counts directly.
- It calls deterministic Python tools on real data.
- Final answer is based on tool outputs.

---

## B) Document Q&A (`POST /api/documents/chat`)

Imagine user uploads a PDF and asks:

> "Summarize this document"

Flow:

1. User uploads file using `POST /api/documents/upload`.
2. `document_service.py` extracts text:
   - PDF: `PyPDF2`
   - DOCX: `python-docx`
3. Text is stored in memory (`DocumentStore`) with `doc_id`.
4. User asks question with `doc_id` at `POST /api/documents/chat`.
5. API builds a system prompt that includes full document text.
6. API calls Claude on Vertex (`AnthropicVertex`).
7. Claude returns answer based on document content.
8. API returns `{ response, doc_id }`.

### Important behavior

- Storage is in-memory (session only).
- Uploaded docs are lost on restart.
- Document Q&A is separate from deficiency DataFrame Q&A.

---

## Step 3: Tool Calling Explained Simply

In this app, **tools** are Python functions that the LLM can call.

- Tool definitions are in `backend/tools.py`.
- Actual logic is in `backend/data_service.py`.

Example:

- Model wants count by market.
- It calls tool `count_by_column(group_by="Markets")`.
- Backend executes function in pandas.
- Result goes back to model.
- Model writes final explanation for user.

So LLM acts as planner/explainer, and tools act as the data engine.

---

## Step 4: Data Engine (`data_service.py`) - What It Does

At startup:

1. Loads Excel (`DATA_FILE_PATH` or `data/deficiency_data.xlsx`).
2. Converts date columns to datetime.
3. Fills nulls for key text fields.

Then it exposes reusable query functions such as:

- `filter_deficiencies`
- `count_by_column`
- `search_descriptions`
- `get_time_series`
- `get_products_under_review`
- `get_response_time_outliers`
- `get_themes_from_descriptions`

These functions are the core source of truth for calculations.

---

## Step 5: Logging, Feedback, and Analytics

`backend/feedback_store.py` stores data in SQLite (`data/feedback.db`).

Two tables:

- `query_logs`
  - query text, response, tools used, execution time, success/failure
- `feedback`
  - user rating/comment linked by `query_id`

Why this matters:

- You can audit what happened for each question.
- You can analyze negative feedback and improve prompts/tools.

---

## Backend API Endpoints (Quick List)

## Core

- `GET /health`
- `POST /chat`

## Data endpoints

- `GET /data/schema`
- `GET /data/summary`
- `POST /data/filter`
- `POST /data/count`
- `POST /data/search`
- `GET /data/unique/{column}`
- `GET /data/time-series`

## Feedback/analytics

- `POST /api/feedback`
- `GET /api/analytics/feedback`
- `GET /api/analytics/queries`
- `GET /api/analytics/suggestions`
- `GET /api/analytics/export`

## Document APIs

- `POST /api/documents/upload`
- `GET /api/documents`
- `DELETE /api/documents/{doc_id}`
- `POST /api/documents/chat`

---

## Test Without curl (Python Only)

Use existing script:

- `test_document_qa.py`

Run:

```bash
source .venv/bin/activate
python test_document_qa.py "/absolute/path/to/file.pdf"
```

This script:
1. uploads the document
2. prints `doc_id`
3. starts interactive terminal chat

---

## Very Short Frontend Note

Frontend is optional for backend testing.

You can test backend directly using:
- FastAPI docs: `http://localhost:8000/docs`
- Python `requests` scripts
- Postman/any HTTP client

---

## Common Issues and Fixes

## 1) `/chat` fails with API key error

- Check `OPENAI_API_KEY` in `.env`.
- Restart backend after changing `.env`.

## 2) Document chat fails on Vertex

- Verify `GOOGLE_APPLICATION_CREDENTIALS` path.
- Verify service account has Vertex permissions.
- Verify exact `CLAUDE_MODEL` snapshot value.

## 3) Data file not loading

- Check `DATA_FILE_PATH`.
- Or place Excel file at `data/deficiency_data.xlsx`.

## 4) Backend starts but hangs on requests

- Check `http://localhost:8000/health` first.
- Check terminal logs for stack trace.
- Ensure only one process is using port 8000.

---

## Final Mental Model

When user asks a question, backend does this:

- receive request -> ask LLM what tools are needed -> run tools on real data -> return human-readable answer + structured results -> log everything.

That is the core design.

