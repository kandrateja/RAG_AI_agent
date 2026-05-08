"""HTTP routes for the DIMS Letters API."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.adapters import get_adapter
from app.adapters.filestore import SUPPORTED_EXTS
from app.api.schemas import (
    AskRequest,
    AskResponse,
    CreateSessionRequest,
    DraftRequest,
    DraftResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LetterList,
    LetterOut,
    SessionOut,
    SessionTurnOut,
    SubmissionList,
    SubmissionOut,
    UploadResponse,
)
from app.config import get_settings
from app.index import LetterRecord, SubmissionRecord, get_index
from app.llm import llm_available
from app.pipeline.ingest import UploadedFile, ingest_from_adapter, ingest_uploaded_files
from app.rag.answer import answer_question
from app.rag.draft import DraftSpec, draft_letter
from app.rag.sessions import Session, get_session_store

logger = logging.getLogger(__name__)
router = APIRouter()


def _letter_out(letter: LetterRecord, sub: SubmissionRecord, request: Request) -> LetterOut:
    base = str(request.base_url).rstrip("/")
    return LetterOut(
        id=letter.id,
        submission_id=letter.submission_id,
        kind=letter.kind,
        filename=letter.filename,
        sha256=letter.sha256,
        page_count=letter.page_count,
        ingested_at=letter.ingested_at,
        download_url=f"{base}/letters/{letter.id}/download",
        text_url=f"{base}/letters/{letter.id}/text",
        product=sub.drug_name,
        anda_number=sub.anda_number,
        ectd_seq=sub.ectd_seq,
    )


def _submission_out(sub: SubmissionRecord, request: Request) -> SubmissionOut:
    index = get_index()
    letters = index.letters_for_submission(sub.id)
    return SubmissionOut(
        id=sub.id,
        source=sub.source,
        source_ref=sub.source_ref,
        folder_path=sub.folder_path,
        anda_number=sub.anda_number,
        ectd_seq=sub.ectd_seq,
        drug_name=sub.drug_name,
        dosage=sub.dosage,
        submit_date=sub.submit_date,
        applicant=sub.applicant,
        sender_name=sub.sender_name,
        ingested_at=sub.ingested_at,
        letters=[_letter_out(l, sub, request) for l in letters],
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    index = get_index()
    return HealthResponse(
        status="ok",
        indexed_submissions=len(index.submissions),
        indexed_letters=len(index.letters),
        llm_configured=llm_available(),
        model=s.anthropic_model if llm_available() else None,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    adapter = get_adapter(req.source)
    stats = ingest_from_adapter(adapter)
    return IngestResponse(**stats.to_dict())


@router.get("/submissions", response_model=SubmissionList)
def list_submissions(
    request: Request,
    anda: str | None = Query(None),
    seq: str | None = Query(None),
    product: str | None = Query(None, description="Substring match on product / drug name"),
) -> SubmissionList:
    index = get_index()
    rows = index.find_submissions(anda=anda, seq=seq, product=product)
    return SubmissionList(
        items=[_submission_out(r, request) for r in rows],
        total=len(rows),
    )


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(submission_id: str, request: Request) -> SubmissionOut:
    sub = get_index().get_submission(submission_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    return _submission_out(sub, request)


@router.get("/letters", response_model=LetterList)
def list_letters(
    request: Request,
    anda: str | None = Query(None),
    seq: str | None = Query(None),
    product: str | None = Query(None, description="Substring match on product / drug name"),
    kind: str | None = Query(None, pattern="^(cover|response|annexure|other)$"),
) -> LetterList:
    index = get_index()
    rows = index.find_letters(anda=anda, seq=seq, product=product, kind=kind)
    return LetterList(
        items=[_letter_out(l, s, request) for l, s in rows],
        total=len(rows),
    )


@router.get("/letters/{letter_id}", response_model=LetterOut)
def get_letter(letter_id: str, request: Request) -> LetterOut:
    index = get_index()
    letter = index.get_letter(letter_id)
    if letter is None:
        raise HTTPException(status_code=404, detail="Letter not found")
    sub = index.get_submission(letter.submission_id)
    if sub is None:
        raise HTTPException(status_code=500, detail="Index inconsistent: missing submission")
    return _letter_out(letter, sub, request)


@router.get("/letters/{letter_id}/text", response_class=PlainTextResponse)
def get_letter_text(letter_id: str) -> str:
    letter = get_index().get_letter(letter_id)
    if letter is None:
        raise HTTPException(status_code=404, detail="Letter not found")
    if not letter.markdown_path:
        raise HTTPException(status_code=404, detail="No extracted text on file")
    p = Path(letter.markdown_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Extracted text missing on disk")
    return p.read_text(encoding="utf-8")


@router.get("/letters/{letter_id}/download")
def download_letter(letter_id: str) -> FileResponse:
    letter = get_index().get_letter(letter_id)
    if letter is None:
        raise HTTPException(status_code=404, detail="Letter not found")
    p = Path(letter.original_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Original file missing on disk")
    media_type = "application/pdf" if p.suffix.lower() == ".pdf" else "application/octet-stream"
    return FileResponse(path=str(p), filename=letter.filename, media_type=media_type)


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return answer_question(
        req.question,
        session_id=req.session_id,
        letter_id=req.letter_id,
    )


def _session_out(s: Session) -> SessionOut:
    return SessionOut(
        id=s.id,
        created_at=s.created_at,
        updated_at=s.updated_at,
        focus_letter_id=s.focus_letter_id,
        focus_submission_id=s.focus_submission_id,
        last_intent=s.last_intent,
        history=[SessionTurnOut(question=t.question, answer=t.answer, at=t.at) for t in s.history],
    )


@router.post("/sessions", response_model=SessionOut)
def create_session(req: CreateSessionRequest | None = None) -> SessionOut:
    """Create a new conversation session.

    Optionally anchor it to an existing letter via ``letter_id`` so the
    next ``/ask`` call answers questions against that letter without the
    user having to name the product/ANDA.
    """
    store = get_session_store()
    focus_letter_id: str | None = None
    focus_submission_id: str | None = None
    if req and req.letter_id:
        letter = get_index().get_letter(req.letter_id)
        if letter is None:
            raise HTTPException(status_code=404, detail="letter_id not found")
        focus_letter_id = letter.id
        focus_submission_id = letter.submission_id
    s = store.create(
        focus_letter_id=focus_letter_id,
        focus_submission_id=focus_submission_id,
    )
    return _session_out(s)


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> SessionOut:
    s = get_session_store().get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return _session_out(s)


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    deleted = get_session_store().delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    return {"deleted": True, "session_id": session_id}


@router.post("/upload", response_model=UploadResponse)
async def upload_letters(
    # NOTE: Swagger UI only renders a proper multi-file picker when the
    # parameter is declared as ``list[UploadFile] = File(...)`` — the
    # Annotated form drops the ``format: binary`` hint and degrades to a
    # plain string textbox. The B008 ruff warning is intentional here.
    files: list[UploadFile] = File(  # noqa: B008
        ..., description="One or more PDF / DOCX letters to ingest."
    ),
    submission_ref: str | None = Form(  # noqa: B008
        None,
        description="Optional logical name for the submission (defaults to a timestamped id).",
    ),
) -> UploadResponse:
    """Upload one or more letter files directly and ingest them.

    All uploaded files are grouped into a single submission and run
    through the full Docling extract -> classify -> index pipeline. A
    fresh conversation session is opened, anchored to the first ingested
    letter so the caller can immediately ``/ask`` follow-up questions
    using the returned ``session_id`` without naming the product.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    payload: list[UploadedFile] = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type {suffix!r} for {f.filename!r}; "
                f"allowed: {sorted(SUPPORTED_EXTS)}",
            )
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Uploaded file {f.filename!r} is empty")
        payload.append(UploadedFile(filename=f.filename or "upload.bin", data=data))

    try:
        result = ingest_uploaded_files(payload, submission_ref=submission_ref)
    except Exception as exc:
        logger.exception("Upload ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    store = get_session_store()
    session = store.create(
        focus_letter_id=result.letter_ids[0] if result.letter_ids else None,
        focus_submission_id=result.submission_id,
    )

    n_created = result.stats.letters_created
    n_skipped = result.stats.letters_skipped
    if n_created == 0 and n_skipped > 0:
        message = (
            f"All {n_skipped} uploaded file(s) were duplicates of letters already in the "
            "knowledge base — re-using the existing entries."
        )
    elif result.stats.errors:
        message = (
            f"Ingested {n_created} new letter(s); {len(result.stats.errors)} file(s) "
            "failed (see errors)."
        )
    else:
        message = f"Ingested {n_created} new letter(s) under submission {result.submission_id!r}."

    return UploadResponse(
        submission_id=result.submission_id,
        letter_ids=result.letter_ids,
        letters_created=n_created,
        letters_skipped=n_skipped,
        errors=list(result.stats.errors),
        session_id=session.id,
        message=message,
    )


@router.post("/draft", response_model=DraftResponse)
def draft(req: DraftRequest) -> DraftResponse:
    """Draft a brand-new letter for a product/scenario, using past letters
    of the same kind as few-shot exemplars. Used when the requested
    letter doesn't exist in the knowledge base yet (e.g. a product
    you're filing for the first time)."""
    spec = DraftSpec(
        kind=req.kind,
        product=req.product,
        anda_number=req.anda_number,
        ectd_seq=req.ectd_seq,
        submit_date=req.submit_date,
        applicant=req.applicant,
        context=req.context,
        instructions=req.instructions,
        max_examples=req.max_examples,
    )
    result = draft_letter(spec)
    return DraftResponse(
        letter=result.letter,
        kind=result.kind,
        used_examples=result.used_examples,
        used_model=result.used_model,
        notes=result.notes,
    )
