"""
Provider-agnostic LLM client.

    from src.llm_client import chat
    response = chat(messages, system=..., tools=..., max_tokens=...)
    response.content      # list of {"type": "text"|"tool_use", ...} dicts
    response.stop_reason  # "end_turn" | "tool_use" | "max_tokens"

The canonical message / tool / response shape is Anthropic's. Adapters convert
to and from each provider's native schema. To swap providers, set LLM_PROVIDER:

    LLM_PROVIDER=anthropic   (default)
    LLM_PROVIDER=openai
    LLM_PROVIDER=gemini

Required env vars per provider:
    anthropic  ANTHROPIC_API_KEY,  optional ANTHROPIC_BASE_URL
    openai     OPENAI_API_KEY,     optional OPENAI_BASE_URL
    gemini     GEMINI_API_KEY (or GOOGLE_API_KEY)

Optional SDKs are imported lazily so users only install what they need:
    pip install anthropic              # default
    pip install openai                 # for LLM_PROVIDER=openai
    pip install google-generativeai    # for LLM_PROVIDER=gemini
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class LLMResponse:
    content: list  # list of dicts, Anthropic-shape
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"
    raw: Any = None


def chat(messages, *, system: str = "", tools: list | None = None,
         max_tokens: int = 8000, model: str | None = None) -> LLMResponse:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = model or os.environ["MODEL_ID"]
    tools = tools or []
    if provider == "anthropic":
        return _anthropic(messages, system, tools, max_tokens, model)
    if provider == "openai":
        return _openai(messages, system, tools, max_tokens, model)
    if provider in ("gemini", "google"):
        return _gemini(messages, system, tools, max_tokens, model)
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'anthropic', 'openai', or 'gemini'."
    )


def _btype(b) -> str | None:
    return b["type"] if isinstance(b, dict) else getattr(b, "type", None)


def _bfield(b, key):
    return b.get(key) if isinstance(b, dict) else getattr(b, key, None)


# --- Anthropic --------------------------------------------------------------

def _anthropic(messages, system, tools, max_tokens, model) -> LLMResponse:
    from anthropic import Anthropic
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    client = Anthropic(base_url=base_url)
    resp = client.messages.create(
        model=model, system=system, messages=messages,
        tools=tools, max_tokens=max_tokens,
    )
    content = []
    for b in resp.content:
        if b.type == "text":
            content.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return LLMResponse(content=content, stop_reason=resp.stop_reason, raw=resp)


# --- OpenAI -----------------------------------------------------------------

def _openai(messages, system, tools, max_tokens, model) -> LLMResponse:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("LLM_PROVIDER=openai requires: pip install openai") from e

    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))

    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    for m in messages:
        oai_messages.extend(_to_openai_msgs(m))

    oai_tools = [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["input_schema"],
        },
    } for t in tools] or None

    resp = client.chat.completions.create(
        model=model, messages=oai_messages, tools=oai_tools, max_tokens=max_tokens,
    )

    choice = resp.choices[0]
    msg = choice.message
    content = []
    if msg.content:
        content.append({"type": "text", "text": msg.content})
    for tc in (msg.tool_calls or []):
        content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.function.name,
            "input": json.loads(tc.function.arguments or "{}"),
        })

    stop_reason = {
        "tool_calls": "tool_use",
        "stop": "end_turn",
        "length": "max_tokens",
    }.get(choice.finish_reason, choice.finish_reason)
    return LLMResponse(content=content, stop_reason=stop_reason, raw=resp)


def _to_openai_msgs(m):
    role, content = m["role"], m["content"]
    if isinstance(content, str):
        return [{"role": role, "content": content}]

    if role == "assistant":
        text_parts, tool_calls = [], []
        for b in content:
            t = _btype(b)
            if t == "text":
                text_parts.append(_bfield(b, "text"))
            elif t == "tool_use":
                tool_calls.append({
                    "id": _bfield(b, "id"),
                    "type": "function",
                    "function": {
                        "name": _bfield(b, "name"),
                        "arguments": json.dumps(_bfield(b, "input") or {}),
                    },
                })
        msg = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    # user role: tool_result blocks become "tool" role messages; text stays user
    out, user_text = [], []
    for b in content:
        t = _btype(b)
        if t == "tool_result":
            out.append({
                "role": "tool",
                "tool_call_id": _bfield(b, "tool_use_id"),
                "content": _bfield(b, "content"),
            })
        elif t == "text":
            user_text.append(_bfield(b, "text"))
    if user_text:
        out.insert(0, {"role": "user", "content": "".join(user_text)})
    return out


# --- Gemini -----------------------------------------------------------------

def _gemini(messages, system, tools, max_tokens, model) -> LLMResponse:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError(
            "LLM_PROVIDER=gemini requires: pip install google-genai"
        ) from e

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for LLM_PROVIDER=gemini")
    client = genai.Client(api_key=api_key)

    name_by_id = _collect_tool_name_map(messages)

    gemini_tools = None
    if tools:
        gemini_tools = [types.Tool(function_declarations=[{
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": _clean_schema(t["input_schema"]),
        } for t in tools])]

    contents = []
    for m in messages:
        contents.extend(_to_gemini_msgs(m, name_by_id))

    config = types.GenerateContentConfig(
        system_instruction=system or None,
        tools=gemini_tools,
        max_output_tokens=max_tokens,
    )
    resp = client.models.generate_content(
        model=model, contents=contents, config=config,
    )

    content, tool_used = [], False
    for part in resp.candidates[0].content.parts:
        fc = getattr(part, "function_call", None)
        sig = getattr(part, "thought_signature", None)
        if fc and getattr(fc, "name", None):
            tool_used = True
            block = {
                "type": "tool_use",
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "name": fc.name,
                "input": dict(fc.args) if fc.args else {},
            }
            if sig:
                block["_gemini_signature"] = sig
            content.append(block)
        elif getattr(part, "text", ""):
            block = {"type": "text", "text": part.text}
            if sig:
                block["_gemini_signature"] = sig
            content.append(block)

    stop_reason = "tool_use" if tool_used else "end_turn"
    return LLMResponse(content=content, stop_reason=stop_reason, raw=resp)


def _collect_tool_name_map(messages):
    name_by_id = {}
    for m in messages:
        c = m.get("content")
        if m["role"] != "assistant" or not isinstance(c, list):
            continue
        for b in c:
            if _btype(b) == "tool_use":
                name_by_id[_bfield(b, "id")] = _bfield(b, "name")
    return name_by_id


def _to_gemini_msgs(m, name_by_id):
    role, content = m["role"], m["content"]
    g_role = "model" if role == "assistant" else "user"

    if isinstance(content, str):
        return [{"role": g_role, "parts": [{"text": content}]}]

    parts = []
    for b in content:
        t = _btype(b)
        sig = _bfield(b, "_gemini_signature")
        if t == "text":
            part = {"text": _bfield(b, "text")}
            if sig:
                part["thought_signature"] = sig
            parts.append(part)
        elif t == "tool_use":
            part = {"function_call": {
                "name": _bfield(b, "name"),
                "args": _bfield(b, "input") or {},
            }}
            if sig:
                part["thought_signature"] = sig
            parts.append(part)
        elif t == "tool_result":
            tool_id = _bfield(b, "tool_use_id")
            parts.append({"function_response": {
                "name": name_by_id.get(tool_id, tool_id),
                "response": {"result": _bfield(b, "content")},
            }})
    return [{"role": g_role, "parts": parts}] if parts else []


def _clean_schema(schema):
    """Strip JSON-schema keys Gemini's function-declaration parser rejects."""
    if not isinstance(schema, dict):
        return schema
    drop = {"$schema", "additionalProperties", "title"}
    out = {k: v for k, v in schema.items() if k not in drop}
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: _clean_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _clean_schema(out["items"])
    return out
