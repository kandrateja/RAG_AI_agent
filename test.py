"""
Countermeasures & Regulatory Response Generator.
Given a theme name and its deficiency descriptions (with product names),
generates a single comprehensive countermeasures document covering root causes,
CAPA, checkpoints, FDA response guidance, and proactive risk mitigation
using Claude on Vertex AI.
"""

import asyncio
import json
import logging
import math
import os
from typing import List

import anthropic
from anthropic import AnthropicVertex
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF_BASE = 2.0
MAX_CHARS_PER_CHUNK = 60_000


def _build_anthropic_vertex_client():
    project_id = os.getenv("VERTEX_PROJECT_ID", "prj-thematic-analysis-dev")
    region = os.getenv("VERTEX_REGION", "us-east5")
    sa_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if sa_key_path:
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = service_account.Credentials.from_service_account_file(sa_key_path, scopes=scopes)
        creds.refresh(Request())
        return AnthropicVertex(region=region, project_id=project_id, credentials=creds)

    return AnthropicVertex(region=region, project_id=project_id)


class CountermeasuresOutput(BaseModel):
    countermeasures: str = Field(
        min_length=50,
        description=(
            "A single comprehensive document containing: "
            "1) Theme Summary, 2) Common Root Causes, "
            "3) Countermeasures (CAPA), 4) Pre-submission Checkpoints, "
            "5) Recommended Response to FDA, 6) Proactive Risk Mitigation."
        ),
    )


SYSTEM_PROMPT = (
    "You are a senior US FDA regulatory affairs expert with 20+ years of experience "
    "in pharmaceutical quality compliance, CDER submissions, and responding to FDA "
    "deficiency letters (Complete Response Letters, Information Requests, and "
    "Refuse-to-File communications)."
)

USER_PROMPT_TEMPLATE = """
Input:
  Theme Name: {theme_name}
  Deficiency Data (product names and deficiency descriptions):
{chunk_text}

You are given:
1. A THEME NAME — representing a cluster of related FDA deficiency observations.
2. Raw data containing PRODUCT NAMES and their corresponding DEFICIENCY DESCRIPTIONS that belong to this theme. Identify the product names and descriptions from the data yourself.

Your task: Analyze ALL the deficiency descriptions under this theme and produce ONE comprehensive regulatory response document.

The document MUST contain ALL of the following sections, clearly separated with markdown headers:

## 1. Theme Summary
- A 2-3 sentence summary of what this theme covers and why it matters to FDA reviewers.

## 2. Common Root Causes
- List the top 3-5 root causes that typically lead to these deficiencies.
- For each root cause, explain WHY the FDA considers it a deficiency.

## 3. Countermeasures (CAPA)
- For each root cause, provide specific corrective and preventive actions.
- Each countermeasure must be actionable — not vague advice like "improve documentation." Instead: "Revise the stability protocol to include accelerated and long-term conditions per ICH Q1A(R2), with justified time points at 0, 3, 6, 9, 12, 18, 24, and 36 months."

## 4. Pre-Submission Checkpoints
- A numbered checklist of items that a regulatory team should verify BEFORE submitting to the FDA to prevent this type of deficiency from recurring.
- Each checkpoint should be specific and verifiable (yes/no).
- Example: "Specification limits for each impurity are justified with toxicological data or ICH Q3A/Q3B limits."

## 5. Recommended Response to FDA
- A template/framework for how to respond if the FDA has already raised this deficiency.
- Include: what data to provide, what justifications to include, what commitments to make, and what language to use.
- The tone should be professional, scientifically rigorous, and compliant.

## 6. Proactive Risk Mitigation
- What can the organization do going forward to prevent this category of deficiencies across ALL future submissions — not just the current one.
- Think: SOPs, training, review checklists, process improvements.

RULES:
1. Be specific to the actual deficiency descriptions provided — do not give generic advice.
2. Cite relevant FDA guidances, ICH guidelines (Q1A, Q1B, Q2, Q3A, Q3B, Q6A, Q8, Q9, Q10, Q11, etc.), and 21 CFR sections where applicable.
3. If a deficiency pattern suggests a systemic issue, call it out explicitly.
4. Prioritize countermeasures by impact — address the most frequently occurring deficiency patterns first.
5. The output must be useful for a regulatory affairs professional to directly act on.
6. Return EVERYTHING as a single string inside the "countermeasures" field.
"""


def _chunk_text(text: str) -> List[str]:
    """Split text into chunks only if it exceeds the size limit."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]

    num_chunks = math.ceil(len(text) / MAX_CHARS_PER_CHUNK)
    lines = text.split("\n")
    lines_per_chunk = max(1, math.ceil(len(lines) / num_chunks))

    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk = "\n".join(lines[i : i + lines_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


async def generate_countermeasures(
    theme_name: str, deficiency_data: str
) -> dict:
    chunks = _chunk_text(deficiency_data)
    if not chunks:
        raise ValueError("deficiency_data is empty")
    print(f"Deficiency data: {len(deficiency_data)} chars → {len(chunks)} chunk(s)")

    client = _build_anthropic_vertex_client()
    tasks = [_analyze_chunk(theme_name, chunk, client) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes: List[CountermeasuresOutput] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            logger.error("Chunk %d/%d failed: %s", i + 1, len(results), r)
        else:
            successes.append(r)

    if not successes:
        raise RuntimeError(
            f"All {len(chunks)} chunk(s) failed. Check logs for details."
        )

    if len(successes) < len(results):
        logger.warning(
            "%d/%d chunks succeeded; generating from partial results",
            len(successes), len(results),
        )

    if len(successes) == 1:
        return successes[0].model_dump()

    return await _consolidate_results(theme_name, successes, client)


async def _call_with_retries(
    client: AnthropicVertex,
    tool_name: str,
    tools: list,
    user_prompt: str,
    temperature: float = 0.2,
    fallback: dict | None = None,
) -> dict | None:
    """Shared retry logic for all Claude API calls."""
    def _call_sync():
        return client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4@20250514"),
            max_tokens=8192,
            temperature=temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
        )

    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.to_thread(_call_sync)
        except anthropic.RateLimitError as e:
            logger.warning("Rate limit (429): attempt %d/%d — %s", attempt + 1, MAX_RETRIES, e)
            if attempt + 1 >= MAX_RETRIES:
                if fallback is not None:
                    return fallback
                raise
            await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
            continue
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600:
                logger.warning("Server error %s: attempt %d/%d", status, attempt + 1, MAX_RETRIES)
                if attempt + 1 >= MAX_RETRIES:
                    if fallback is not None:
                        return fallback
                    raise
                await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise
        except anthropic.APITimeoutError as e:
            logger.warning("Timeout: attempt %d/%d — %s", attempt + 1, MAX_RETRIES, e)
            if attempt + 1 >= MAX_RETRIES:
                if fallback is not None:
                    return fallback
                raise
            await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
            continue

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
                return getattr(block, "input", None)

        text = "".join(
            getattr(b, "text", "") for b in response.content
            if getattr(b, "type", "") == "text"
        )
        try:
            return json.loads(text)
        except Exception:
            if fallback is not None:
                return fallback
            raise ValueError("Model did not return a tool_call and no valid JSON found in text.")

    if fallback is not None:
        return fallback
    raise RuntimeError("All retry attempts exhausted")


def _safe_parse(result) -> CountermeasuresOutput:
    """Convert whatever Claude returned into a CountermeasuresOutput, no matter the shape."""
    # Happy path: result already has the correct key
    if isinstance(result, dict) and "countermeasures" in result:
        try:
            return CountermeasuresOutput.model_validate(result)
        except Exception:
            pass

    # Claude used a different key name — grab the longest string value
    if isinstance(result, dict):
        longest = ""
        for v in result.values():
            if isinstance(v, str) and len(v) > len(longest):
                longest = v
        if longest:
            return CountermeasuresOutput(countermeasures=longest)

        # Dict of sub-sections — join them all into one document
        parts = []
        for k, v in result.items():
            if isinstance(v, str) and v.strip():
                header = k.replace("_", " ").title()
                parts.append(f"## {header}\n{v.strip()}")
        if parts:
            return CountermeasuresOutput(countermeasures="\n\n".join(parts))

    # Result is a plain string (rare, but possible from fallback JSON parse)
    if isinstance(result, str) and len(result) >= 50:
        return CountermeasuresOutput(countermeasures=result)

    # None or something unexpected — stringify as last resort
    raw = json.dumps(result, indent=2) if result is not None else ""
    if len(raw) >= 50:
        return CountermeasuresOutput(countermeasures=raw)

    raise ValueError(f"Could not extract countermeasures from response: {type(result)}")


async def _analyze_chunk(
    theme_name: str, chunk_text: str, client: AnthropicVertex
) -> CountermeasuresOutput:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        theme_name=theme_name, chunk_text=chunk_text
    )

    tool_name = "submit_analysis"
    tools = [
        {
            "name": tool_name,
            "description": "Return the regulatory countermeasures document.",
            "input_schema": CountermeasuresOutput.model_json_schema(),
        }
    ]

    result = await _call_with_retries(client, tool_name, tools, user_prompt)
    return _safe_parse(result)


async def _consolidate_results(
    theme_name: str,
    chunk_results: List[CountermeasuresOutput],
    client: AnthropicVertex,
) -> dict:
    """Merge multiple chunk analyses into one consolidated document."""
    all_parts = "\n\n---\n\n".join(r.countermeasures for r in chunk_results)
    fallback = {"countermeasures": all_parts}

    consolidation_prompt = f"""
You are given multiple partial countermeasures analyses for the FDA deficiency theme "{theme_name}".
Each section below was generated from a different subset of deficiency descriptions.

Merge them into ONE cohesive, deduplicated final document. Remove redundancy,
keep the most actionable and specific points, and ensure the final output reads as a
single unified document with all sections (Theme Summary, Common Root Causes,
Countermeasures/CAPA, Checkpoints, Recommended Response to FDA, Proactive Risk Mitigation).

Return everything in the single "countermeasures" field.

PARTIAL RESULTS:

{all_parts}
"""

    tool_name = "submit_consolidated"
    tools = [
        {
            "name": tool_name,
            "description": "Return the consolidated regulatory countermeasures document.",
            "input_schema": CountermeasuresOutput.model_json_schema(),
        }
    ]

    result = await _call_with_retries(
        client, tool_name, tools, consolidation_prompt,
        temperature=0.1, fallback=fallback,
    )
    if result is fallback:
        return fallback

    try:
        validated = _safe_parse(result)
        return validated.model_dump()
    except Exception as e:
        logger.warning("Consolidation parse failed (%s); using concatenated fallback", e)
        return fallback
