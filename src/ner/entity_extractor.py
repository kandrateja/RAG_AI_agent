"""
Entity and Relationship Extraction using LLM (Azure OpenAI or Bedrock Claude).
"""
import logging
import json
import re
from typing import List, Dict, Optional, Any
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


def _repair_json(text: str) -> str:
    """Attempt to repair common JSON issues from LLM output. Handles Unicode (Arabic, etc.)."""
    # Remove any text before the first { and after the last }
    match = re.search(r'\{.*\}', text, re.DOTALL | re.UNICODE)
    if match:
        text = match.group(0)
    
    # Fix trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Fix single quotes to double quotes (but not apostrophes in words)
    # Use \w+ for keys (works with ASCII keys), but be careful with values that may contain Unicode
    text = re.sub(r"'(\w+)':", r'"\1":', text)  # 'key': -> "key":
    # For values: match single-quoted strings, but preserve Unicode content
    # This regex matches ': 'value' followed by comma/brace/bracket
    text = re.sub(r":\s*'([^']*)'([,}\]])", r': "\1"\2', text, flags=re.UNICODE)  # : 'value' -> : "value"
    
    return text


def _salvage_json(text: str) -> Optional[Dict]:
    """
    On parse failure, try to salvage a valid JSON prefix by truncating at error
    and closing brackets. Returns None if not possible.
    """
    try:
        decoder = json.JSONDecoder()
        obj, idx = decoder.raw_decode(text)
        return obj
    except json.JSONDecodeError as e:
        if e.pos is None or e.pos <= 0:
            return None
        # Try truncating at error position and close any open structures
        truncated = text[: e.pos]
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        suffix = ']' * open_brackets + '}' * open_braces
        if open_braces < 0 or open_brackets < 0:
            return None
        try:
            return json.loads(truncated + suffix)
        except json.JSONDecodeError:
            return None


class EntityExtractor:
    """Extract entities and relationships from text using an LLM (Azure OpenAI or Bedrock)."""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        *,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize entity extractor.
        If llm_client is provided (e.g. BedrockClient), it is used for completions;
        otherwise Azure OpenAI is used.
        llm_client must provide: chat_completion(messages, max_completion_tokens=...) -> str
        """
        self._llm_client = llm_client
        self.client = None if llm_client else AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
    
    def extract_entities_and_relationships(
        self,
        text: str,
        max_entities: int = 25
    ) -> Dict:
        """
        Extract entities and relationships from text
        
        Args:
            text: Input text to analyze
            max_entities: Maximum number of entities to extract
            
        Returns:
            Dictionary with entities and relationships
        """
        try:
            # Normalize input so the model always sees real text (avoid leading whitespace / empty slices)
            text_in = (text or "").strip()
            if not text_in:
                print("[ENTITY] Entity extraction skipped: empty input text")
                logger.info("Entity extraction skipped: empty input text")
                return {"entities": [], "relationships": []}

            # Send more text for Arabic documents (they may need more context)
            text_sample = text_in[:6000] if len(text_in) > 4000 else text_in
            print(f"[ENTITY] Input chars: {len(text_in)} (sending {len(text_sample)} chars)")
            print(f"[ENTITY] First 100 chars: {text_sample[:100]!r}")
            logger.info(f"Entity extraction input chars: {len(text_in)} (sending {len(text_sample)} chars, first 100: {text_sample[:100]!r})")

            # Stricter prompt for better JSON output; explicitly support Arabic and other languages
            prompt = f"""Extract named entities and relationships from this text. Return ONLY a valid JSON object.

TEXT (between <BEGIN_TEXT> and <END_TEXT>):
<BEGIN_TEXT>
{text_sample}
<END_TEXT>

INSTRUCTIONS:
- The text may be in English, Arabic, or mixed. Extract entities and relationships in any language.
- IMPORTANT: If the text contains Arabic, extract Arabic entities. Arabic text is valid in JSON string values.
- Keep entity "name" and "description" in the original language (e.g. Arabic names stay in Arabic, English stay in English).
- Use type in English: Person, Organization, Location, Concept, Date, Event, etc.
- Extract up to {min(max_entities, 25)} important entities.
- Extract relationships between those entities ("from" and "to" must be exact entity names as in the entities list).
- Use double quotes for all strings. Escape any quotes inside strings with backslash.
- JSON supports Unicode (Arabic characters are valid in JSON strings).
- No trailing commas. No comments or explanations. Output only the JSON object.

OUTPUT FORMAT (return ONLY this JSON, nothing else):
{{"entities":[{{"name":"string (can be Arabic)","type":"string","description":"string (can be Arabic)"}}],"relationships":[{{"from":"string","to":"string","type":"string","description":"string"}}]}}

EXAMPLE (Arabic text):
{{"entities":[{{"name":"محمد","type":"Person","description":"اسم شخص"}}],"relationships":[]}}"""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only entity extraction assistant. "
                        "Extract entities and relationships from text in any language (e.g. English, Arabic, mixed). "
                        "JSON supports Unicode - Arabic characters are valid in JSON string values. "
                        "Output valid JSON only: no markdown, no code fences, no explanatory text before or after. "
                        "If the text is Arabic, extract Arabic entities and return them in JSON with Arabic strings."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            if self._llm_client is not None:
                result_text = self._llm_client.chat_completion(
                    messages=messages,
                    max_completion_tokens=2000,
                    temperature=0.1,  # Lower temperature for more deterministic output
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=2000,
                    temperature=0.1,
                )
                result_text = response.choices[0].message.content.strip()

            result_text = (result_text or "").strip()
            print(f"[ENTITY] LLM response length: {len(result_text)} chars")
            print(f"[ENTITY] LLM response (first 300): {result_text[:300]!r}")
            if not result_text:
                print("[ENTITY] Empty response from LLM")
                logger.warning("Entity extraction got empty response from LLM; skipping entities/relationships")
                return {"entities": [], "relationships": []}

            # Try to extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result_text = result_text.strip()
            if not result_text:
                logger.warning("Entity extraction: no JSON found inside response (empty after stripping code blocks)")
                return {"entities": [], "relationships": []}

            # First attempt: parse as-is
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                repaired = _repair_json(result_text)
                try:
                    result = json.loads(repaired)
                except json.JSONDecodeError as e:
                    # Third attempt: salvage valid prefix (e.g. truncate at error position)
                    result = _salvage_json(result_text)
                    if result is None:
                        result = _salvage_json(repaired)
                    if result is None:
                        raise e
            
            entities = result.get("entities", [])[:50]  # cap in case of huge output
            relationships = result.get("relationships", [])[:100]
            
            print(f"[ENTITY] SUCCESS: Extracted {len(entities)} entities, {len(relationships)} relationships")
            logger.info(f"Extracted {len(entities)} entities, {len(relationships)} relationships")
            
            return {
                "entities": entities,
                "relationships": relationships
            }
            
        except json.JSONDecodeError as e:
            print(f"[ENTITY] JSON PARSE ERROR: {e}")
            logger.warning(f"Failed to parse entity extraction JSON: {e}")
            try:
                preview = (result_text[:800] + "..." if len(result_text) > 800 else result_text)
                print(f"[ENTITY] Raw response: {preview!r}")
                logger.warning(f"Entity extraction raw response (first 800 chars): {preview!r}")
            except NameError:
                pass
            return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"[ENTITY] ERROR: {str(e)}")
            logger.error(f"Error extracting entities: {str(e)}")
            return {"entities": [], "relationships": []}
