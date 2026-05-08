"""Extract structured fields from a cover letter's markdown body.

Two-stage strategy:

1. Regex: cheap, deterministic, handles 95% of FDA cover letters which
   follow a stable header layout (ANDA #, eCTD Seq #, drug + dosage,
   ``Submit On:`` line, ``Sincerely,`` block).
2. LLM fallback (Anthropic Claude Opus on Vertex AI, mirroring the
   ``dims_automation`` project) only for fields the regex stage could
   not recover. Optional — gracefully no-ops if Vertex isn't configured.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime

logger = logging.getLogger(__name__)


@dataclass
class CoverFields:
    anda_number: str | None = None
    ectd_seq: str | None = None
    drug_name: str | None = None
    dosage: str | None = None
    submit_date: date | None = None
    applicant: str | None = None
    sender_name: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.submit_date is not None:
            d["submit_date"] = self.submit_date.isoformat()
        return d


_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_date(s: str) -> date | None:
    s = s.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


_RE_ANDA = re.compile(r"ANDA\s*(?:Number)?\s*#?\s*(\d{4,8})", re.IGNORECASE)
_RE_SEQ = re.compile(r"eCTD\s*Seq\.?\s*#?\s*(\d{1,6})", re.IGNORECASE)
# "Submit On:" may sit on its own line with the date in the next block.
_RE_SUBMIT = re.compile(
    r"Submit\s*On\s*:?\s*\n*\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4})",
    re.IGNORECASE,
)
# Drug + dosage anywhere in the body (Docling doesn't keep a single drug line).
_DRUG_FORMS = (
    "Capsules",
    "Tablets",
    "Injection",
    "Solution",
    "Suspension",
    "Cream",
    "Ointment",
    "Gel",
    "Spray",
    "Drops",
    "Tab",
    "Cap",
    "Powder",
)
_RE_DRUG_LINE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:" + "|".join(_DRUG_FORMS) + r"))"
    r"\s*,?\s*([\d.]+\s*(?:mg|mcg|g|ml|%))",
)
_RE_NAME = re.compile(r"^([A-Z][\w'\-\.]+(?:\s+[A-Z][\w'\-\.]+){1,3})\s*,")


def _extract_sender(markdown: str) -> tuple[str | None, str | None]:
    """Find the applicant block + the named designee right after 'Sincerely,'."""
    m = re.search(r"Sincerely,?\s*(.+)\Z", markdown, re.DOTALL | re.IGNORECASE)
    if not m:
        return None, None
    block = m.group(1).strip()
    # Split on blank lines first; each paragraph is a logical block.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
    applicant: str | None = None
    sender_name: str | None = None

    for para in paragraphs:
        raw_first = para.splitlines()[0].strip()
        if applicant is None:
            applicant = raw_first.rstrip(",")
            continue
        # Look for "Firstname Lastname," at the start of the paragraph -> sender.
        if (mn := _RE_NAME.match(raw_first)):
            sender_name = mn.group(1).strip()
            break
    return applicant, sender_name


def extract_cover_fields_regex(markdown: str) -> CoverFields:
    fields = CoverFields()

    if (m := _RE_ANDA.search(markdown)):
        fields.anda_number = m.group(1)
    if (m := _RE_SEQ.search(markdown)):
        fields.ectd_seq = m.group(1).zfill(4)
    if (m := _RE_SUBMIT.search(markdown)):
        fields.submit_date = _parse_date(m.group(1))
    if (m := _RE_DRUG_LINE.search(markdown)):
        fields.drug_name = f"{m.group(1).strip()} {m.group(2).strip()}"
        fields.dosage = m.group(2).strip()

    fields.applicant, fields.sender_name = _extract_sender(markdown)
    return fields


def is_cover_letter(filename: str, markdown: str) -> bool:
    fname = filename.lower()
    if "cover" in fname:
        return True
    head = markdown[:2000].lower()
    return "submit on:" in head and "anda" in head and "dear sir" in head


def classify_letter_kind(filename: str, markdown: str) -> str:
    fn = filename.lower()
    # Order matters: annexure / response file names can also contain the
    # word "cover" (e.g. "annexure-to-cover-letter.pdf"). Check those
    # specific kinds first.
    if "annex" in fn or "exhibit" in fn or "appendix" in fn:
        return "annexure"
    if "response" in fn or "reply" in fn:
        return "response"
    if is_cover_letter(filename, markdown):
        return "cover"
    return "other"


def llm_fill_missing(markdown: str, fields: CoverFields) -> CoverFields:
    """Use Claude on Vertex to fill any unset fields. No-op if Vertex isn't configured."""
    from app.llm import create_message, llm_available

    if not llm_available():
        return fields

    missing = [k for k, v in fields.to_dict().items() if v is None]
    if not missing:
        return fields

    system = (
        "You extract structured metadata from FDA regulatory cover letters. "
        "Return STRICT JSON with exactly the requested keys (use null when unknown). "
        "submit_date must be ISO YYYY-MM-DD."
    )
    user = (
        f"Requested keys: {missing}\n\n"
        "Letter markdown:\n---\n"
        f"{markdown[:8000]}\n---\n"
        "Return only the JSON object, no prose, no markdown fences."
    )
    try:
        raw = create_message(system=system, user_content=user, max_tokens=512, temperature=0.0)
        data = json.loads(_strip_json_fences(raw))
    except Exception as exc:
        logger.warning("LLM field fill failed: %s", exc)
        return fields

    for key in missing:
        val = data.get(key)
        if not val:
            continue
        if key == "submit_date":
            d = _parse_date(str(val))
            if d:
                fields.submit_date = d
        else:
            setattr(fields, key, str(val).strip() or None)
    return fields


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def extract_cover_fields(markdown: str, *, use_llm_fallback: bool = True) -> CoverFields:
    fields = extract_cover_fields_regex(markdown)
    if use_llm_fallback:
        fields = llm_fill_missing(markdown, fields)
    return fields
