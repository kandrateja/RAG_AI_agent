"""
Amazon Bedrock LLM client for Claude (text + vision).
Used for chat, vision captions, and entity extraction when Bedrock is configured.
"""
import json
import logging
from types import SimpleNamespace
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO_AVAILABLE = True
except ImportError:
    BOTO_AVAILABLE = False
    boto3 = None  # type: ignore


class BedrockClient:
    """
    Claude on Amazon Bedrock (Anthropic messages API).
    Supports text chat and image (vision) inputs via base64.
    """

    def __init__(
        self,
        region_name: str,
        model_id: str,
        max_tokens: int = 4096,
        *,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        if not BOTO_AVAILABLE:
            raise RuntimeError("boto3 is not installed. Install with: pip install boto3")
        self.region_name = region_name
        self.model_id = model_id
        self.max_tokens = max_tokens
        kwargs: Dict[str, Any] = {"service_name": "bedrock-runtime", "region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key_id
            kwargs["aws_secret_access_key"] = aws_secret_access_key
        self._client = boto3.client(**kwargs)

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate chat completion (Anthropic messages format).
        messages: [{"role": "user"|"assistant", "content": str or list of content blocks}]
        system: Optional system prompt (passed as top-level parameter, not in messages)
        
        Note: If messages contain role="system", they are extracted and combined into
        the system parameter automatically (Bedrock doesn't accept system as a message role).
        """
        max_tokens = max_completion_tokens or self.max_tokens
        
        # Extract system messages and normalize the rest
        system_prompt, normalized_messages = self._normalize_messages(messages)
        
        # Combine explicit system param with extracted system messages
        if system and system_prompt:
            system_prompt = f"{system}\n\n{system_prompt}"
        elif system:
            system_prompt = system
        
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": normalized_messages,
            "temperature": temperature,
        }
        
        # Add system prompt as top-level parameter (not in messages)
        if system_prompt:
            body["system"] = system_prompt
        
        if kwargs:
            body.update(kwargs)

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            content = result.get("content", [])
            if not content:
                return ""
            # Collect all text from blocks (some models return thinking then text; we need the final text)
            text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            return "".join(text_parts).strip() if text_parts else content[0].get("text", "").strip()
        except (BotoCoreError, ClientError, KeyError, IndexError) as e:
            logger.error(f"Bedrock chat_completion error: {e}")
            raise

    def _normalize_messages(self, messages: List[Dict]) -> tuple:
        """
        Extract system messages and normalize the rest.
        Returns: (system_prompt: str or None, normalized_messages: List[Dict])
        
        Bedrock only accepts "user" and "assistant". We extract "system" and
        convert "tool" messages to "user" with tool_result content blocks.
        """
        system_parts = []
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m.get("role", "user")
            content = m.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            system_parts.append(block)
                i += 1
                continue

            if role == "tool":
                # Bedrock does not accept role "tool". Send as user with tool_result blocks.
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tm = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tm.get("tool_call_id", ""),
                        "content": tm.get("content", "") if isinstance(tm.get("content"), str) else json.dumps(tm.get("content", "")),
                    })
                    i += 1
                out.append({"role": "user", "content": tool_results})
                continue

            if role == "assistant" and m.get("tool_calls"):
                # Anthropic expects assistant content as tool_use blocks, not a separate field.
                blocks = []
                for tc in m.get("tool_calls", []):
                    if isinstance(tc, dict):
                        tid = tc.get("id", "")
                        fn = tc.get("function") or {}
                        fname = fn.get("name", "") if isinstance(fn, dict) else ""
                        args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                    else:
                        tid = getattr(tc, "id", "")
                        fn = getattr(tc, "function", None)
                        fname = getattr(fn, "name", "") if fn else ""
                        args = getattr(fn, "arguments", "{}") if fn else "{}"
                    try:
                        inp = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        inp = {}
                    blocks.append({"type": "tool_use", "id": tid, "name": fname, "input": inp})
                out.append({"role": "assistant", "content": blocks})
                i += 1
                continue

            # user or assistant (plain content)
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": ""}]
            out.append({"role": role, "content": content})
            i += 1

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, out

    def chat_completion_with_image(
        self,
        prompt: str,
        image_base64: str,
        max_completion_tokens: Optional[int] = None,
        media_type: str = "image/png",
        **kwargs: Any,
    ) -> str:
        """
        Vision: single user turn with one image (base64) and text prompt.
        """
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        return self.chat_completion(
            messages,
            max_completion_tokens=max_completion_tokens or 256,
            **kwargs,
        )

    def chat_completion_raw(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        max_completion_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Tool-use completion: same semantics as OpenAI chat_completion_raw.
        Returns an object with response.choices[0].message and message.tool_calls
        (each with .id, .function.name, .function.arguments).
        """
        max_tokens = max_completion_tokens or self.max_tokens

        # Convert OpenAI-style tools to Anthropic format
        bedrock_tools: List[Dict[str, Any]] = []
        if tools:
            for t in tools:
                fn = (t.get("function") or t) if isinstance(t.get("function"), dict) else {}
                name = fn.get("name", "")
                desc = fn.get("description", "")
                params = fn.get("parameters") or {"type": "object", "properties": {}}
                bedrock_tools.append({
                    "name": name,
                    "description": desc or f"Tool: {name}",
                    "input_schema": params,
                })

        # Convert tool_choice: OpenAI {"type": "function", "function": {"name": "X"}} -> Anthropic {"type": "tool", "name": "X"}
        bedrock_tool_choice: Any = "auto"
        if tool_choice and bedrock_tools:
            if isinstance(tool_choice, dict):
                fn_part = tool_choice.get("function") or {}
                name = fn_part.get("name") if isinstance(fn_part, dict) else None
                if name:
                    bedrock_tool_choice = {"type": "tool", "name": name}

        # Convert messages for Anthropic (assistant with tool_calls -> content tool_use; tool -> user tool_result)
        system_prompt, anthropic_messages = self._normalize_messages_for_tools(messages)

        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "temperature": kwargs.get("temperature", 0.7),
        }
        if system_prompt:
            body["system"] = system_prompt
        if bedrock_tools:
            body["tools"] = bedrock_tools
            body["tool_choice"] = bedrock_tool_choice

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
        except (BotoCoreError, ClientError, KeyError) as e:
            logger.error(f"Bedrock chat_completion_raw error: {e}")
            raise

        # Map Anthropic response to OpenAI-like shape: choices[0].message with tool_calls
        content_blocks = result.get("content", [])
        tool_calls_list: List[Any] = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                inp = block.get("input") or {}
                args_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
                tc = SimpleNamespace(
                    id=block.get("id", ""),
                    type="function",  # OpenAI-style; _tool_call_to_dict expects call.type
                    function=SimpleNamespace(
                        name=block.get("name", ""),
                        arguments=args_str,
                    ),
                )
                tool_calls_list.append(tc)

        msg = SimpleNamespace(tool_calls=tool_calls_list)
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])

    def _normalize_messages_for_tools(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages (including assistant with tool_calls and role=tool)
        to Anthropic format. Returns (system_prompt, anthropic_messages).
        """
        system_parts = []
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            m = messages[i]
            role = m.get("role", "user")
            content = m.get("content")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_parts.append(block.get("text", ""))
                i += 1
                continue

            if role == "tool":
                # Collect consecutive tool messages into one user message with tool_result blocks
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tm = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tm.get("tool_call_id", ""),
                        "content": tm.get("content", "") if isinstance(tm.get("content"), str) else json.dumps(tm.get("content", "")),
                    })
                    i += 1
                out.append({"role": "user", "content": tool_results})
                continue

            if role == "assistant" and m.get("tool_calls"):
                # Convert tool_calls to Anthropic content blocks (tool_use)
                blocks = []
                for tc in m.get("tool_calls", []):
                    if isinstance(tc, dict):
                        tid = tc.get("id", "")
                        fn = tc.get("function") or {}
                        fname = fn.get("name", "") if isinstance(fn, dict) else ""
                        args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                    else:
                        tid = getattr(tc, "id", "")
                        fn = getattr(tc, "function", None)
                        fname = getattr(fn, "name", "") if fn else ""
                        args = getattr(fn, "arguments", "{}") if fn else "{}"
                    try:
                        inp = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        inp = {}
                    blocks.append({"type": "tool_use", "id": tid, "name": fname, "input": inp})
                out.append({"role": "assistant", "content": blocks})
                i += 1
                continue

            # user or assistant (plain content)
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                content = [{"type": "text", "text": ""}]
            out.append({"role": role, "content": content})
            i += 1

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, out
