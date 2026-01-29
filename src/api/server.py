"""
FastAPI server wrapping the RAG AI Agent.

Exposes HTTP endpoints for:
- /health
- /ingest
- /query
"""
import sys
from pathlib import Path

# Ensure project root is on path so "config" and "src" resolve when running uvicorn from any cwd
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import tempfile
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_agent import RAGAgent
import logging

# Configure logging to show detailed application logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Also set specific loggers to INFO
for log_name in ['src', 'src.rag_agent', 'src.ner', 'src.ner.entity_extractor', 'src.ingestion', '__main__']:
    logging.getLogger(log_name).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG AI Agent API", version="1.0.0")

# Allow local dev by default; tighten in production as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI assets
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: Optional[str] = None
    page_number: Optional[int] = None
    similarity: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    content_type: Optional[str] = "text"  # "text" or "image"
    source_label: Optional[str] = None  # "Figure 1", etc. for images


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: Optional[str] = None
    page_number: Optional[int] = None
    content: Optional[str] = None
    similarity: Optional[float] = None
    content_type: Optional[str] = "text"
    image_b64: Optional[str] = None  # Base64 image data for image chunks


class WebCitation(BaseModel):
    web_id: str
    request_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
class ChatMessage(BaseModel):
    role: str
    content: str



class IngestResponse(BaseModel):
    doc_id: str
    status: str
    chunks_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_graph_context: bool = True
    history: Optional[List[ChatMessage]] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    web_citations: Optional[List[WebCitation]] = None
    provenance: str  # "internal", "online", "both", "none", "error"
    tools_used: dict
    tools_satisfied: Optional[dict] = None
    sources_used: Optional[dict] = None
    decision_trace: Optional[dict] = None
    retrieved_chunks: List[RetrievedChunk] = []  # Full chunk data including images
    retrieved_chunks_count: int
    web_results_count: int
    has_internal_knowledge: bool
    internal_sufficient: bool
    entities_count: int = 0
    relationships_count: int = 0
    error: Optional[str] = None


# Single global agent instance for the API process
agent: Optional[RAGAgent] = None


@app.on_event("startup")
def on_startup() -> None:
    global agent
    try:
        logger.info("Initializing RAG Agent...")
        agent = RAGAgent()
        logger.info("RAG Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Agent: {e}")
        raise


@app.on_event("shutdown")
def on_shutdown() -> None:
    global agent
    if agent is not None:
        logger.info("Shutting down RAG Agent...")
        agent.close()
        agent = None


@app.get("/health")
def health() -> dict:
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent_initialized": agent is not None
    }


@app.get("/")
def ui_root() -> FileResponse:
    """Serve the UI HTML."""
    return FileResponse("ui/index.html")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
    pipeline_hint: Optional[str] = Form(None),
) -> IngestResponse:
    """
    Ingest a document via file upload.

    The file is temporarily saved to disk, passed to the RAGAgent,
    and then removed.
    
    pipeline_hint: optional. Default "auto" - Arabic and handwritten/scanned are auto-detected; no need to pass.
    Supports deduplication - re-ingesting the same document will be skipped.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save to a temporary file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        logger.info(f"Ingesting document: {file.filename} (pipeline_hint={pipeline_hint})")
        result = agent.ingest_document(
            tmp_path, doc_id=doc_id, source_name=file.filename, pipeline_hint=pipeline_hint
        )
        
        status = result.get("status", "unknown")
        
        if status == "error":
            error_type = result.get("error_type", "unknown_error")
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Ingestion failed: {error_msg} (type: {error_type})")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": error_msg,
                    "error_type": error_type
                }
            )
        
        return IngestResponse(
            doc_id=result.get("doc_id", ""),
            status=status,
            chunks_created=result.get("chunks_created", 0),
            entities_extracted=result.get("entities_extracted", 0),
            relationships_extracted=result.get("relationships_extracted", 0),
            message=result.get("message"),
            error=result.get("error"),
            error_type=result.get("error_type")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            # Best-effort cleanup
            pass


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system.

    The agent will decide automatically whether to use:
    - graph (GraphRAG / Neo4j),
    - vector-only,
    - web search (Surf-like API),
    or a combination of these tools based on the question.
    
    Returns:
    - answer: The generated answer
    - citations: List of citations with doc_id, page_number, chunk_id
    - provenance: Source of answer ("internal", "online", "both", "none", "error")
    - tools_used: Which tools were used
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        result = agent.query(
            question=request.question,
            top_k=request.top_k,
            use_graph_context=request.use_graph_context,
            history=[m.dict() for m in request.history] if request.history else None,
        )

        # Format citations
        citations = []
        for cit in result.get("citations", []):
            citations.append(Citation(
                chunk_id=cit.get("chunk_id", "unknown"),
                doc_id=cit.get("doc_id", "unknown"),
                doc_name=cit.get("doc_name"),
                page_number=cit.get("page_number"),
                similarity=cit.get("similarity"),
                semantic_score=cit.get("semantic_score"),
                keyword_score=cit.get("keyword_score"),
                content_type=cit.get("content_type", "text"),
                source_label=cit.get("source_label"),
            ))

        # Format retrieved chunks (including image data for frontend display)
        retrieved_chunks = []
        for chunk in result.get("retrieved_chunks", []):
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk.get("chunk_id", "unknown"),
                doc_id=chunk.get("doc_id", "unknown"),
                doc_name=chunk.get("doc_name"),
                page_number=chunk.get("page_number"),
                content=chunk.get("content"),
                similarity=chunk.get("similarity"),
                content_type=chunk.get("content_type", "text"),
                image_b64=chunk.get("image_b64"),  # Include image data
            ))

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            citations=citations,
            web_citations=result.get("web_citations"),
            provenance=result.get("provenance", "none"),
            tools_used=result.get("tools_used", {}),
            tools_satisfied=result.get("tools_satisfied"),
            sources_used=result.get("sources_used"),
            decision_trace=result.get("decision_trace"),
            retrieved_chunks=retrieved_chunks,
            retrieved_chunks_count=len(retrieved_chunks),
            web_results_count=len(result.get("web_results", [])),
            has_internal_knowledge=result.get("has_internal_knowledge", False),
            internal_sufficient=result.get("internal_sufficient", False),
            entities_count=len(result.get("entities", [])),
            relationships_count=len(result.get("relationships", [])),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# For local debugging: `python -m src.api.server`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
