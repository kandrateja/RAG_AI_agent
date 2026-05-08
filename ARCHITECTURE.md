# DIMS Letters API — Architecture

A layered walkthrough of the system, scaling from a 30-second overview for non-technical readers down to a component-level view for engineers. Every Mermaid block below renders as an image directly on GitHub.

---

## 1. The 30-second summary

> **DIMS Letters API** lets a user **upload an FDA letter (PDF / DOCX) and chat with it in plain English** — or ask the system to **draft a new letter** in Dr. Reddy's house style based on past letters. It runs on FastAPI, uses **IBM Docling** to read PDFs, and **Anthropic Claude Opus on Google Vertex AI** to understand and write text. No database — just a JSON index file plus the extracted markdown on disk.

```mermaid
flowchart LR
    U["👤 User<br/>(browser, curl, app)"]
    A["📬 DIMS Letters API"]
    R["📚 Read the letter<br/>(IBM Docling)"]
    B["🧠 Understand & answer<br/>(Claude Opus on Vertex AI)"]
    K["💾 Knowledge Base<br/>(JSON index +<br/>letter markdown)"]
    O["💬 Answer / drafted letter<br/>+ session id for follow-ups"]

    U -->|"1. Upload letter<br/>or ask question"| A
    A --> R
    A --> K
    R --> K
    A -->|"3. Generate answer<br/>or new draft"| B
    K --> B
    B --> O
    O --> U
    U -.->|"2. Follow-up Qs<br/>same session id"| A

    classDef user fill:#fde68a,stroke:#b45309,color:#3f2d05;
    classDef core fill:#bfdbfe,stroke:#1d4ed8,color:#0b1f4d;
    classDef ai fill:#fecaca,stroke:#b91c1c,color:#3f0712;
    classDef store fill:#ddd6fe,stroke:#6d28d9,color:#1e1346;
    classDef out fill:#bbf7d0,stroke:#15803d,color:#062b13;
    class U,O user;
    class A,R core;
    class B ai;
    class K store;
```

**Read it like a story:**

1. User uploads a letter or asks a question.
2. The API reads the PDF (Docling) and stores the text in a small knowledge base.
3. Claude Opus answers the question — or drafts a brand-new letter — using past letters as examples.
4. The user gets the answer and can keep asking follow-ups via a session id.

---

## 2. Two user journeys (step-by-step)

### Journey A — *"I have a letter. Let me ask questions about it."*

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as DIMS API
    participant Docling
    participant KB as JSON index +<br/>markdown on disk
    participant Sess as Session store<br/>(in-memory)
    participant Claude as Claude Opus<br/>on Vertex AI

    User->>API: POST /upload (PDF)
    API->>Docling: extract text
    Docling-->>API: markdown + page count
    API->>KB: save letter + metadata
    API->>Sess: open session, anchor to letter
    API-->>User: { session_id, letter_id, ... }

    User->>API: POST /ask { question, session_id }
    API->>Sess: load focus letter + last intent
    API->>KB: read letter markdown
    API->>Claude: question + letter + history
    Claude-->>API: natural-language answer
    API->>Sess: append turn (Q, A)
    API-->>User: { answer, session_id, mode: "focus" }

    Note over User,API: Follow-ups reuse session_id —<br/>"Who signed it?" / "When was it submitted?"<br/>still resolve to the same letter
```

### Journey B — *"I don't have the letter yet — draft one for me."*

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as DIMS API
    participant KB as JSON index +<br/>markdown on disk
    participant Drafter as Few-shot drafter
    participant Claude as Claude Opus<br/>on Vertex AI

    User->>API: POST /ask "Draft a cover letter for<br/>Atorvastatin 10 mg, ANDA #999999"
    API->>API: parse intent → wants_draft=true
    API->>KB: pick past cover letters as examples<br/>(same product first, then most recent)
    KB-->>API: 3-6 example letters
    API->>Drafter: build few-shot prompt
    Drafter->>Claude: system + exemplars + new spec
    Claude-->>API: brand-new letter in house style
    API-->>User: { answer (the letter), used_examples, mode: "draft" }
```

---

## 3. Technical architecture (for engineers)

Five clear layers, top to bottom. Each box has only its role; the table after the diagram maps boxes to actual files.

```mermaid
flowchart TB
    subgraph L1["1️⃣  Clients"]
        UI["Swagger UI<br/>:8000/docs"]
        CURL["curl / SDK / app"]
    end

    subgraph L2["2️⃣  API Layer · FastAPI (app/api)"]
        UP["POST /upload"]
        ING["POST /ingest"]
        ASK["POST /ask"]
        DR["POST /draft"]
        SESS["/sessions  (POST · GET · DELETE)"]
        BROWSE["/letters  ·  /submissions  ·  /health"]
    end

    subgraph L3["3️⃣  Business Logic"]
        direction LR
        PIPE["📥 Ingestion Pipeline<br/>(app/pipeline)<br/>extract → classify → fields → index"]
        RAG["🧠 RAG Layer<br/>(app/rag)<br/>intent · answer · draft · sessions"]
    end

    subgraph L4["4️⃣  Storage"]
        IDX[("JSON Index<br/>data/index.json")]
        BLOB[("Blob Root<br/>data/blobs/")]
        MEM[("Session Store<br/>in-memory, TTL 6h")]
    end

    subgraph L5["5️⃣  External"]
        DOC["IBM Docling<br/>(local library)"]
        VTX["Google Vertex AI"]
        CL["Anthropic Claude Opus 4"]
    end

    UI --> L2
    CURL --> L2

    UP --> PIPE
    ING --> PIPE
    ASK --> RAG
    DR --> RAG
    SESS --> RAG
    BROWSE --> IDX
    BROWSE --> BLOB

    PIPE --> DOC
    PIPE --> IDX
    PIPE --> BLOB
    RAG --> IDX
    RAG --> BLOB
    RAG --> MEM
    RAG --> VTX
    PIPE -. "optional LLM fallback<br/>for missing fields" .-> VTX
    VTX --> CL

    classDef l1 fill:#fde68a,stroke:#b45309,color:#3f2d05;
    classDef l2 fill:#bfdbfe,stroke:#1d4ed8,color:#0b1f4d;
    classDef l3 fill:#bbf7d0,stroke:#15803d,color:#062b13;
    classDef l4 fill:#ddd6fe,stroke:#6d28d9,color:#1e1346;
    classDef l5 fill:#fecaca,stroke:#b91c1c,color:#3f0712;
    class UI,CURL l1;
    class UP,ING,ASK,DR,SESS,BROWSE l2;
    class PIPE,RAG l3;
    class IDX,BLOB,MEM l4;
    class DOC,VTX,CL l5;
```

### File map (what lives where)

| Layer | Component | Code location | One-line role |
|---|---|---|---|
| API | Routes | `app/api/routes.py` | All HTTP endpoints |
| API | Schemas | `app/api/schemas.py` | Request / response Pydantic models |
| API | App entry | `app/main.py` | FastAPI app; OpenAPI 3.0 downgrade so Swagger renders the upload picker |
| Pipeline | Adapters | `app/adapters/` | Pluggable inputs (file-store, RDB stub) |
| Pipeline | Orchestrator | `app/pipeline/ingest.py` | Folder ingest + direct upload |
| Pipeline | Extract | `app/pipeline/extract.py` | Docling PDF → markdown + page map |
| Pipeline | Fields | `app/pipeline/fields.py` | Classify letter kind, pull header fields (regex + LLM fallback) |
| RAG | Intent parser | `app/rag/intent.py` | Question → `{product, kind, anda, seq, wants_draft}` |
| RAG | Answer | `app/rag/answer.py` | Lookup / focus / disambiguate / draft fall-through |
| RAG | Drafter | `app/rag/draft.py` | Few-shot prompt build, exemplar selection |
| RAG | Sessions | `app/rag/sessions.py` | Multi-turn memory (focus letter, intent carry-over, history) |
| Storage | Index | `app/index.py` → `data/index.json` | All submission + letter records |
| Storage | Blobs | `data/blobs/originals/`, `data/blobs/markdown/` | Original files + Docling output |
| LLM | Vertex client | `app/llm.py` | `AnthropicVertex` (Claude Opus 4 on Vertex) |
| Config | Settings | `app/config.py` + `.env` | All env-driven knobs |

---

## 4. The modes `/ask` can return (cheat sheet)

| `mode` | When | What you get back |
|---|---|---|
| `lookup` | Question matched exactly one indexed letter | Claude's summary + the matched letter |
| `focus` | Same session asked a follow-up about its anchored letter | Answer against the focused letter |
| `disambiguate` | Multiple letters match (e.g. *"everything for Voclosporin"*) | List of candidates; ask the user to narrow down |
| `draft` | User said "draft" / no letter exists for that product | A newly drafted letter in house style |
| `not_found` | Not enough info to look up *or* draft | Friendly error explaining what's missing |

---

## 5. Configuration knobs

All driven by environment variables (or a `.env` file at repo root):

| Variable | Purpose |
|---|---|
| `SOURCES__FILESTORE__ROOT` | Folder root for batch ingest (`POST /ingest`) |
| `BLOB_ROOT` | Where Docling output + originals are persisted (default `data/blobs`) |
| `INDEX_PATH` | Path to the JSON index (default `data/index.json`) |
| `VERTEX_PROJECT_ID` | GCP project hosting Vertex AI |
| `VERTEX_REGION` | Vertex region (default `us-east5`) |
| `ANTHROPIC_MODEL` | Claude model id (default `claude-opus-4@20250514`) |
| `ANTHROPIC_MAX_TOKENS` | Per-call cap (default 4096) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service-account JSON (ADC) |
| `LOG_LEVEL` | Python logging level (default `INFO`) |

If `VERTEX_PROJECT_ID` is unset, the API still works for ingest, upload, listing, and verbatim letter retrieval — only the LLM-powered answers and drafts degrade gracefully with a clear "LLM not configured" message.
