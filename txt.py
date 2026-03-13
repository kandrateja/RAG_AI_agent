"""
Countermeasures & Regulatory Response Generator.
Given a theme name and its deficiency descriptions (with product names),
generates a comprehensive countermeasures document using Claude on Vertex AI.

Architecture: ONE API call with all data (fits in Claude's 1M token context).
Speed optimization: concise output constraints + low max_tokens.
"""

import asyncio
import logging
import os
import time

import anthropic
from anthropic import AsyncAnthropicVertex

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

def _build_async_client() -> AsyncAnthropicVertex:
    region = os.getenv("VERTEX_REGION", "us-east5")
    project_id = os.getenv("VERTEX_PROJECT_ID")
    if not project_id:
        raise ValueError("VERTEX_PROJECT_ID must be set in .env")
    return AsyncAnthropicVertex(region=region, project_id=project_id)


def _get_model() -> str:
    return os.getenv("COUNTERMEASURE_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4@20250514"))


# ---------------------------------------------------------------------------
# Prompts — optimized for concise, fast output
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a senior US FDA regulatory affairs expert. "
    "You write concise, actionable regulatory guidance in markdown. "
    "No JSON, no code blocks, no preamble — only the document."
)

ANALYSIS_PROMPT = """Theme: {theme_name}

Deficiency data:
{deficiency_data}

Analyze ALL deficiency descriptions above. Produce a COUNTERMEASURES document with EXACTLY these 4 sections.
Be specific to the actual deficiency descriptions — cite ICH/FDA guidances and 21 CFR sections. No generic advice.
Prioritize by frequency of occurrence. Keep each section focused — aim for 5-10 key points per section, not exhaustive lists.

## Countermeasures (CAPA)
Specific corrective and preventive actions for the top issue patterns found in the data.
Each must be actionable with concrete steps (not "improve documentation" but "Revise protocol to include X per ICH Y").

## Pre-Submission Checkpoints
Numbered yes/no checklist items a regulatory team should verify before submission.

## Recommended Response to FDA
Template for responding if FDA already raised this deficiency. Include what data/justifications to provide.

## Proactive Risk Mitigation
Forward-looking org-wide improvements: SOPs, training, review checklists, process changes.
"""


# ---------------------------------------------------------------------------
# Core async API call with retries
# ---------------------------------------------------------------------------

async def _call_claude(
    client: AsyncAnthropicVertex,
    user_prompt: str,
    max_tokens: int = 4096,
) -> str:
    model = _get_model()

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.RateLimitError as e:
            logger.warning("Rate limit (429): attempt %d/%d — %s", attempt + 1, MAX_RETRIES, e)
            if attempt + 1 >= MAX_RETRIES:
                raise
            await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
            continue
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600:
                logger.warning("Server error %s: attempt %d/%d", status, attempt + 1, MAX_RETRIES)
                if attempt + 1 >= MAX_RETRIES:
                    raise
                await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise
        except anthropic.APITimeoutError as e:
            logger.warning("Timeout: attempt %d/%d — %s", attempt + 1, MAX_RETRIES, e)
            if attempt + 1 >= MAX_RETRIES:
                raise
            await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))
            continue

        text = response.content[0].text.strip()
        if text:
            return text

        logger.warning("Empty response; retrying (%d/%d)", attempt + 1, MAX_RETRIES)
        if attempt + 1 >= MAX_RETRIES:
            raise ValueError("Model returned empty response after all retries.")
        await asyncio.sleep(BACKOFF_BASE ** (attempt + 1))

    raise RuntimeError("All retry attempts exhausted")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_countermeasures(
    theme_name: str, deficiency_data: str
) -> dict:
    if not deficiency_data.strip():
        raise ValueError("deficiency_data is empty")

    client = _build_async_client()
    prompt = ANALYSIS_PROMPT.format(
        theme_name=theme_name,
        deficiency_data=deficiency_data,
    )

    input_chars = len(deficiency_data)
    max_tokens = int(os.getenv("COUNTERMEASURE_MAX_TOKENS", "4096"))
    print(f"Input: {input_chars} chars | max_tokens: {max_tokens} | model: {_get_model()}")

    t0 = time.time()
    text = await _call_claude(client, prompt, max_tokens=max_tokens)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s | Output: {len(text)} chars")

    return {"countermeasures": text}
