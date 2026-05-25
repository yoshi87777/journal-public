"""life-os :: insight engine — shared seam between the live server and .team runner.

ONE place that owns:
  1. The system prompt for the insight role (sourced from .team/agents/insight.md).
  2. The response schema (sourced from .team/schemas/insight.response.json).
  3. The provider-agnostic LLM call (via vendor/llm_client.py).
  4. Defensive output shaping (truncate strings, enforce caps, drop unknown keys).

Callers:
  - journal/scripts/feedback_server.py — on every POST /api/journal.
  - .team/run.py role=insight — when running offline against the inbox.

LLM_PROVIDER selection (anthropic | openai | gemini) is read at call time from
the environment. The matching API key must be present. If none works, callers
should fall back to feedback_rules.build_feedback().

This module never raises on LLM failure — it returns None, and lets the caller
decide what to do (typically: fall back to rules).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Vendored provider-agnostic chat(). Lazy-loaded so unrelated imports don't blow up.
_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


# Bootstrap this project's .env so direct imports of this module also get the
# right LLM_PROVIDER / MODEL_ID / API keys. Mirrors feedback_server's loader.
def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    life_os = _HERE.parent.parent
    env_path = life_os / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


_bootstrap_env()

# Project roots.
_LIFE_OS = _HERE.parent.parent
_TEAM = _LIFE_OS / ".team"
_AGENT_PROMPT_PATH = _TEAM / "agents" / "insight.md"


# ---- prompt + schema (cached at import time) ------------------------------

def _read_text(p: Path, fallback: str) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return fallback


_SYSTEM_PROMPT_FALLBACK = (
    "You are a language-aware writing companion AND a substantive insight "
    "engine for the writer. Match the language the writer is practicing in the "
    "entry; if it is unclear, default to English. The writer does NOT want basic "
    "upgrades.\n\n"
    "LANGUAGE CHANNEL (cap 3, can be 0): Target polished language. SKIP:\n"
    "  - 'very + adj' swaps (he knows this)\n"
    "  - simple contractions ('do not' → 'don't')\n"
    "  - elementary word swaps ('really tired' → 'wiped')\n"
    "  - anything a basic textbook would teach\n"
    "INSTEAD, target:\n"
    "  - subtle preposition tuning ('anxious for' → 'anxious about')\n"
    "  - collocational nuance ('make a discussion' → 'have a discussion')\n"
    "  - register tuning (casual / business / academic)\n"
    "  - idiomatic phrasal verbs ('I will give him this info' → 'I'll fill him in')\n"
    "  - removing L2 wordiness ('in order to understand' → 'to make sense of')\n"
    "  - elegant constructions where the original is grammatically fine but plain\n"
    "  - L2-coded patterns ('I have a question to ask' → 'I have a question')\n"
    "  - hedge calibration (over-hedging vs native softening)\n"
    "  - article + tense subtleties when they're load-bearing\n"
    "If the writer's writing for a given sentence is already idiomatic, return zero "
    "language upgrades for it. Cap at 3 strongest across the whole entry. "
    "Note field should explain the nuance briefly — not lecture.\n\n"
    "REFLECT CHANNEL (cap 3, can be 0): substantive observations — themes, "
    "echoes, recurring threads. Never moralize. Never advise. Observations only.\n\n"
    "Output ONLY valid JSON. No prose around it."
)

SYSTEM_PROMPT = _read_text(_AGENT_PROMPT_PATH, _SYSTEM_PROMPT_FALLBACK)

# A small example so the model knows the exact shape. Kept inline so we don't
# have to teach the model JSON Schema dialect.
_SHAPE_EXAMPLE = """{
  "request_id": "<echo of input>",
  "english": [
    {
      "original": "I have been busy these days and could not catch up",
      "suggested": "I've barely surfaced lately — couldn't catch up",
      "note": "'barely surfaced' is a richer idiom for sustained overload; dash + em-em cadence is how writers vary rhythm mid-sentence.",
      "context": "longtext 'free'",
      "kind": "idiom"
    },
    {
      "original": "I am anxious for the presentation tomorrow",
      "suggested": "I'm anxious about the presentation tomorrow",
      "note": "'anxious for' = eager / looking forward (positive). 'anxious about' = nervous (negative). Common L2 slip — preposition flips the meaning.",
      "kind": "word-choice"
    }
  ],
  "reflect": [
    {
      "observation": "you used 'should' 5 times today — a tell.",
      "kind": "should-count",
      "evidence": []
    },
    {
      "observation": "this echoes your may 8 entry about Sugita-san.",
      "kind": "echo",
      "evidence": [{ "source_date": "2026-05-08", "quote": "Sugita-san framed risk differently" }]
    }
  ],
  "gamification": { "streak": 0, "color_count": 0, "new_badge": null },
  "patterns": [
    "'tired' appears 3x this week — try 'wiped' or 'spent'"
  ],
  "prompt_for_tomorrow": "tomorrow: timeline only, no longform.",
  "meta": {
    "source": "gemini",
    "model_id": "gemini-2.5-flash",
    "latency_ms": 0
  }
}"""


# ---- public API -----------------------------------------------------------

def build_insight(
    entry: dict,
    history: list[dict],
    channels: list[str] | None = None,
    *,
    max_tokens: int = 1536,
) -> dict | None:
    """Run the insight role against an entry. Return a shaped dict or None on any failure."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    key = _provider_key(provider)
    if not key:
        # No key for the selected provider — caller will fall back to rules.
        return None

    channels = channels or ["english", "reflect", "patterns", "prompt"]
    # Truncate history for the prompt — past 30 entries, longtext capped at 800 chars each.
    trimmed_history = _trim_history(history, limit=30, longtext_cap=800)

    user_prompt = _build_user_prompt(entry, trimmed_history, channels)

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=[{"role": "user", "content": user_prompt}],
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="insight",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    raw = _extract_text(response)
    parsed = _safe_parse_json(raw)
    if parsed is None:
        return None

    shaped = _shape_response(parsed, provider=provider, model=_resolve_model(provider), latency_ms=latency_ms)
    return shaped


# ---- prompt assembly ------------------------------------------------------

def _build_user_prompt(entry: dict, history: list[dict], channels: list[str]) -> str:
    return (
        "You will produce a single JSON object — no prose, no markdown fences.\n\n"
        f"Channels requested: {', '.join(channels)}\n\n"
        "Today's entry (JSON):\n"
        f"{json.dumps(entry, ensure_ascii=False)}\n\n"
        "Recent history (newest first, trimmed):\n"
        f"{json.dumps(history, ensure_ascii=False)}\n\n"
        "Output JSON shape (match exactly, drop fields you have nothing for):\n"
        f"{_SHAPE_EXAMPLE}\n\n"
        "Rules:\n"
        "- english: 0-3 language-upgrade items (cap). Idiom/word-choice/contraction over grammar.\n"
        "- reflect: 0-3 items (cap). Observations only — never advice.\n"
        "- patterns: 0-3 short strings. Phrase as observations.\n"
        "- prompt_for_tomorrow: one short lowercase sentence.\n"
        "- gamification: leave streak/color_count as null or 0 — the server computes them.\n"
        "- meta.source: use the actual provider name.\n"
        "Empty is honest. If today's writing is already idiomatic, return english=[]. "
        "If there's nothing substantive to reflect on, return reflect=[]."
    )


def _trim_history(history: list[dict], *, limit: int, longtext_cap: int) -> list[dict]:
    out: list[dict] = []
    for h in history[:limit]:
        if not isinstance(h, dict):
            continue
        item: dict[str, Any] = {"date": h.get("date")}
        accent = h.get("accent")
        if isinstance(accent, dict):
            item["accent"] = {"name": accent.get("name")}
        # collect oneliner + longtext from questions if not already top-level
        oneliner = h.get("oneliner")
        longtext = h.get("longtext")
        if not (oneliner or longtext):
            qs = h.get("questions") or []
            longtexts: list[str] = []
            for q in qs:
                if not isinstance(q, dict):
                    continue
                qid = q.get("id")
                v = q.get("value")
                if not isinstance(v, str):
                    continue
                if qid == "oneliner" and not oneliner:
                    oneliner = v
                elif q.get("type") in ("longtext", "shorttext"):
                    longtexts.append(v)
            if longtexts and not longtext:
                longtext = " | ".join(longtexts)
        if oneliner:
            item["oneliner"] = oneliner[:200]
        if longtext:
            item["longtext"] = longtext[:longtext_cap]
        # tags
        tags: list[str] = []
        qs2 = h.get("questions") or []
        for q in qs2:
            if isinstance(q, dict) and q.get("type") == "tags" and isinstance(q.get("value"), list):
                tags.extend([str(t)[:24] for t in q["value"]][:10])
        if tags:
            item["tags"] = tags
        out.append(item)
    return out


# ---- response parsing + shaping ------------------------------------------

def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts).strip()


def _safe_parse_json(raw: str) -> dict | None:
    """Robustly extract a JSON object from a model response.

    Models (especially gemini-flash) often wrap JSON in prose or markdown
    fences. We:
      1. Strip ```json``` / ``` fences.
      2. Try json.loads on the result.
      3. If that fails, locate the outermost balanced {...} substring and
         try again.
    """
    if not raw:
        return None

    # Strip ``` fences anywhere they appear.
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # First try: direct parse.
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: locate the outermost balanced {...}, respecting strings.
    start = s.find("{")
    if start < 0:
        print("[life-os/insight] response had no JSON object", file=sys.stderr)
        return None

    depth = 0
    in_str = False
    esc = False
    end = -1
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end < 0:
        print("[life-os/insight] response had unbalanced braces", file=sys.stderr)
        return None

    candidate = s[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        print(f"[life-os/insight] extracted block was not valid JSON: {e}", file=sys.stderr)
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


_ALLOWED_KINDS_ENGLISH = {"word-choice", "idiom", "contraction", "grammar"}
_ALLOWED_KINDS_REFLECT = {
    "echo", "repetition", "hedge", "should-count", "rhythm",
    "tone-shift", "theme", "open-question", "other",
}


def _shape_response(parsed: dict, *, provider: str, model: str, latency_ms: int) -> dict:
    """Defensively shape the LLM output to match insight.response.json."""

    english_out: list[dict] = []
    eng_in = parsed.get("english")
    if isinstance(eng_in, list):
        for item in eng_in[:3]:
            if not isinstance(item, dict):
                continue
            original = str(item.get("original", "")).strip()[:500]
            suggested = str(item.get("suggested", "")).strip()[:500]
            if not original or not suggested:
                continue
            kind = str(item.get("kind", "word-choice")).strip().lower()
            if kind not in _ALLOWED_KINDS_ENGLISH:
                kind = "word-choice"
            entry_item = {
                "original": original,
                "suggested": suggested,
                "note": str(item.get("note", "")).strip()[:500],
                "kind": kind,
            }
            ctx = item.get("context")
            if isinstance(ctx, str) and ctx.strip():
                entry_item["context"] = ctx.strip()[:120]
            english_out.append(entry_item)

    reflect_out: list[dict] = []
    refl_in = parsed.get("reflect")
    if isinstance(refl_in, list):
        for item in refl_in[:3]:
            if not isinstance(item, dict):
                continue
            obs = str(item.get("observation", "")).strip()[:500]
            if not obs:
                continue
            kind = str(item.get("kind", "other")).strip().lower()
            if kind not in _ALLOWED_KINDS_REFLECT:
                kind = "other"
            r = {"observation": obs, "kind": kind}
            ev_in = item.get("evidence")
            if isinstance(ev_in, list) and ev_in:
                ev_out = []
                for e in ev_in[:3]:
                    if not isinstance(e, dict):
                        continue
                    quote = str(e.get("quote", "")).strip()[:240]
                    src = str(e.get("source_date", "")).strip()[:32]
                    if quote or src:
                        ev_out.append({"source_date": src, "quote": quote})
                if ev_out:
                    r["evidence"] = ev_out
            reflect_out.append(r)

    patterns_out: list[str] = []
    p_in = parsed.get("patterns")
    if isinstance(p_in, list):
        for p in p_in[:3]:
            if isinstance(p, str) and p.strip():
                patterns_out.append(p.strip()[:240])

    tomorrow = parsed.get("prompt_for_tomorrow")
    if isinstance(tomorrow, str):
        tomorrow = tomorrow.strip()[:400]
    else:
        tomorrow = ""

    return {
        "english": english_out,
        "reflect": reflect_out,
        "patterns": patterns_out,
        "prompt_for_tomorrow": tomorrow,
        "meta": {
            "source": provider,
            "model_id": model,
            "latency_ms": latency_ms,
        },
    }


# ---- provider plumbing ----------------------------------------------------

_PROVIDER_KEY_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai":    ("OPENAI_API_KEY",),
    "gemini":    ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google":    ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _provider_key(provider: str) -> str | None:
    # For gemini: also count GEMINI_API_KEYS_EXTRA as "has a key configured".
    for k in _PROVIDER_KEY_VARS.get(provider, ()):
        v = os.environ.get(k)
        if v:
            return v
    if provider in ("gemini", "google"):
        extra = os.environ.get("GEMINI_API_KEYS_EXTRA") or ""
        for token in _re_split_keys(extra):
            return token
    return None


# --- Key rotation for Gemini ---
# the writer configures backup keys in GEMINI_API_KEYS_EXTRA (comma- or
# whitespace-separated) in case the primary hits a quota/rate cap. We try keys
# in order until one works or all are exhausted.
import re as _re_keys


def _re_split_keys(s: str) -> list[str]:
    return [t.strip() for t in _re_keys.split(r"[,\s]+", s or "") if t.strip()]


def _all_gemini_keys() -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = (os.environ.get(env_name) or "").strip()
        if v and v not in seen:
            keys.append(v)
            seen.add(v)
    for k in _re_split_keys(os.environ.get("GEMINI_API_KEYS_EXTRA") or ""):
        if k not in seen:
            keys.append(k)
            seen.add(k)
    return keys


# Process-lifetime cache of keys we've seen get rate-limited.
_EXHAUSTED_KEYS: set[str] = set()
# Quota/rate-limit signatures we treat as "try the next key" rather than fail hard.
_QUOTA_SIGNATURES = (
    "quota", "rate limit", "rate_limit", "ratelimit",
    "429", "resource_exhausted", "resourceexhausted",
    "too many requests", "exceeded",
)
_TRANSIENT_SIGNATURES = (
    "503", "unavailable", "service unavailable",
    "504", "gateway", "timeout",
    "502", "bad gateway",
    "high demand", "try again",
)


def _is_quota_error(exc: Exception) -> bool:
    s = (str(exc) or "").lower()
    if any(sig in s for sig in _QUOTA_SIGNATURES):
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return code == 429


def _is_transient_error(exc: Exception) -> bool:
    """5xx or 'try again later'-shaped errors. Worth a retry, but don't burn keys."""
    s = (str(exc) or "").lower()
    if any(sig in s for sig in _TRANSIENT_SIGNATURES):
        return True
    code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return code in (502, 503, 504)


def _chat_with_retry(
    *,
    messages: list[dict],
    system: str,
    max_tokens: int,
    model: str,
    label: str = "call",
):
    """Wrap llm_client.chat() with Gemini key rotation.

    For non-Gemini providers: single call, single failure point.
    For Gemini: walk through every configured key, skipping ones already
    marked exhausted this process. On 429/quota, mark the key exhausted and
    try the next. On any other error, raise (don't burn through keys for
    a bug or a bad prompt).
    """
    try:
        from llm_client import chat  # vendored
    except ImportError as e:
        print(f"[life-os/insight] llm_client not importable: {e}", file=sys.stderr)
        return None

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider not in ("gemini", "google"):
        return chat(messages=messages, system=system, max_tokens=max_tokens, model=model)

    keys = _all_gemini_keys()
    if not keys:
        return None

    last_err: Exception | None = None
    transient_retries_left = 2  # short-circuit on transient errors, don't loop forever
    for idx, k in enumerate(keys):
        if k in _EXHAUSTED_KEYS:
            continue
        # Surface the key to the SDK without ever logging its value.
        os.environ["GEMINI_API_KEY"] = k
        os.environ["GOOGLE_API_KEY"] = k
        try:
            return chat(messages=messages, system=system, max_tokens=max_tokens, model=model)
        except Exception as e:
            masked = f"key #{idx+1}/{len(keys)}"
            if _is_quota_error(e):
                _EXHAUSTED_KEYS.add(k)
                print(
                    f"[life-os/insight] {label}: gemini quota on {masked} "
                    f"({type(e).__name__}) — rotating",
                    file=sys.stderr,
                )
                last_err = e
                continue
            if _is_transient_error(e):
                if transient_retries_left > 0:
                    transient_retries_left -= 1
                    # Don't mark the key exhausted — the model server was hot.
                    # Brief sleep, try next key (may hit a different backend).
                    print(
                        f"[life-os/insight] {label}: transient error on {masked} "
                        f"({type(e).__name__}) — retrying on next key",
                        file=sys.stderr,
                    )
                    time.sleep(0.4)
                    last_err = e
                    continue
                # Budget spent. Stop trying but don't raise — let the caller
                # fall back to rules. The model is likely globally overloaded
                # (e.g. 503 on a freshly-launched model id); rotating keys
                # won't help, and rules give the writer SOMETHING instead of
                # nothing.
                print(
                    f"[life-os/insight] {label}: transient retries exhausted; "
                    f"falling back to rules ({type(e).__name__})",
                    file=sys.stderr,
                )
                last_err = e
                break
            # Non-recoverable error (auth, schema, bad request): bail.
            raise
    if last_err:
        print(
            f"[life-os/insight] {label}: all gemini keys exhausted or transient "
            f"({len(keys)} tried, last={type(last_err).__name__})",
            file=sys.stderr,
        )
    return None


_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-7",
    "openai":    "gpt-4o",
    "gemini":    "gemini-2.5-flash",
    "google":    "gemini-2.5-flash",
}


def _resolve_model(provider: str) -> str:
    return os.environ.get("MODEL_ID") or _DEFAULT_MODELS.get(provider, "")


# ---- conversation reply ---------------------------------------------------
# When the user replies to a feedback item ("you used 'should' 5 times — a tell"),
# we want a SHORT, conversational follow-up — not another full insight payload.
# Different prompt, different output shape, same provider abstraction.

_REPLY_SYSTEM_PROMPT = (
    "You are continuing a short conversation with the writer about his journal. "
    "He just replied to a feedback observation. Respond in 1-3 sentences. "
    "Lowercase, terse, dry voice (like the rest of the tool). NEVER moralize. "
    "NEVER give advice unless explicitly asked. If he's pushing back on the "
    "observation, take him seriously — your observation might have been wrong, "
    "or partial. If he's expanding on it, reflect what you heard in 1 sentence "
    "and ask at most one clarifying question. Empty turns are OK — if there's "
    "nothing useful to add, say 'noted.' and stop. Output ONLY the assistant's "
    "reply text — no JSON envelope, no labels."
)


def build_conversation_reply(
    *,
    entry: dict | None,
    insight_snapshot: dict | None,
    anchor: dict | None,
    turns: list[dict],
    user_text: str,
    max_tokens: int = 512,
) -> dict | None:
    """Generate a short conversational follow-up.

    Returns:
        {"text": <reply string>, "meta": {"source", "model_id", "latency_ms"}}
        or None if no LLM is available.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not _provider_key(provider):
        return None

    # Build the chat history. Map our turns to the user/assistant roles
    # the chat() abstraction expects. Skip the system-injected snapshot turn
    # — that goes in the system prompt instead.
    prior_messages: list[dict] = []

    # Anchor context (what insight the user is replying to) as the first user-msg framing.
    anchor_text = ""
    if anchor and isinstance(anchor, dict):
        ch = anchor.get("channel")
        snap = anchor.get("snapshot")
        if isinstance(snap, str) and snap.strip():
            anchor_text = f"the user is replying to a {ch} observation: \"{snap.strip()[:300]}\""

    # Today's entry snapshot — give the model a little context but trim.
    entry_snip = ""
    if isinstance(entry, dict):
        qs = entry.get("questions") or []
        if isinstance(qs, list):
            for q in qs:
                if not isinstance(q, dict):
                    continue
                v = q.get("value")
                if isinstance(v, str) and q.get("type") in ("longtext", "shorttext"):
                    entry_snip += f" [{q.get('id')}] {v[:300]}"
            entry_snip = entry_snip.strip()[:1200]

    # Pull a tiny digest of the day's other insights so the model knows
    # what else is on the page besides the anchored item.
    other_insights_snip = ""
    if isinstance(insight_snapshot, dict):
        reflect_items = insight_snapshot.get("reflect") or []
        english_items = insight_snapshot.get("english") or []
        bullets: list[str] = []
        for r in reflect_items[:3]:
            if isinstance(r, dict):
                obs = r.get("observation")
                if isinstance(obs, str):
                    bullets.append(f"reflect: {obs[:160]}")
        for e in english_items[:3]:
            if isinstance(e, dict):
                orig = e.get("original")
                sug = e.get("suggested")
                if isinstance(orig, str) and isinstance(sug, str):
                    bullets.append(f"english: {orig[:80]} → {sug[:80]}")
        if bullets:
            other_insights_snip = "\n".join("- " + b for b in bullets[:6])

    system_context = SYSTEM_PROMPT_FOR_REPLY()
    if anchor_text or entry_snip or other_insights_snip:
        system_context += "\n\nContext:\n"
        if anchor_text:
            system_context += f"- {anchor_text}\n"
        if entry_snip:
            system_context += f"- today's entry text: {entry_snip}\n"
        if other_insights_snip:
            system_context += f"- other observations made today:\n{other_insights_snip}\n"

    # Now thread the turns into the chat history. Anchor is already in system.
    # We replay the full thread including the just-appended user turn.
    for t in turns[-12:]:  # cap to last 12 turns to keep prompt small
        if not isinstance(t, dict):
            continue
        role = t.get("role")
        text = t.get("text")
        if not isinstance(text, str) or role not in ("user", "assistant"):
            continue
        prior_messages.append({"role": role, "content": text})

    # If for some reason the last turn isn't the user_text, append it.
    if not prior_messages or prior_messages[-1].get("content") != user_text:
        prior_messages.append({"role": "user", "content": user_text})

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=prior_messages,
        system=system_context,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="reply",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    reply_text = _extract_text(response).strip()
    if not reply_text:
        return None
    # Strip stray code fences.
    if reply_text.startswith("```"):
        lines = reply_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        reply_text = "\n".join(lines).strip()

    return {
        "text": reply_text[:4000],
        "meta": {
            "source": provider,
            "model_id": _resolve_model(provider),
            "latency_ms": latency_ms,
        },
    }


def SYSTEM_PROMPT_FOR_REPLY() -> str:  # noqa: N802 — small wrapper for clarity
    """Allow easy override of the reply system prompt without editing the constant."""
    return _REPLY_SYSTEM_PROMPT


# ---- question generation --------------------------------------------------
# When the static QBANK has been over-rolled, the frontend asks the LLM for
# fresh questions tailored to today. Output is the same shape as a QBANK item,
# so the frontend can drop it straight into its render path.

_QGEN_SYSTEM_PROMPT = (
    "You generate fresh journal questions for the writer. Match the language the "
    "writer is practicing; if unclear, default to English. Your job: make questions that "
    "make him WANT to write. Inspiring means provocative, specific, weight-bearing. "
    "Never trite. Never 'How are you feeling today?'. Always something that "
    "surfaces an angle he didn't have a name for. Lowercase except proper nouns. "
    "No emoji except in emoji-type widgets. Output ONLY valid JSON matching "
    "the schema you're given — no prose, no fences. Cap totals per the 'want' "
    "block. Vary widget types. Heavies should be writing prompts a thoughtful "
    "person would copy down."
)

_QGEN_GOOD_EXAMPLES = """Good labels — these are what 'inspiring' looks like:

light/slider:
  "How sharp was your edge today?" (min 0 max 10, "dull" / "razor")
  "How heavy did time feel?" (min 0 max 10, "weightless" / "leaden")
  "How loud was your inner monologue?" (0-100, "silent" / "constant")

medium/choice:
  "Today was a ___" : ["sprint", "spiral", "slow walk", "free fall"]
  "When the day is forgotten, what should remain?" : (shorttext)
  "Mark the texture of today" : ["smooth", "rough", "static", "splintered", "humid"]

medium/shorttext:
  "What did you almost not do today, but did?"
  "What did today give you that you didn't expect?"
  "Name a moment you wanted to slow down."
  "What surfaces did you spend yourself on?"
  "Whose voice did you carry today?"

heavy/longtext:
  "Write the part of today that doesn't fit into a one-liner."
  "Describe today the way a film about you would open the scene."
  "The compliment you almost gave today — write it as a letter."
  "What did you defend today, and was it worth defending?"
  "Describe a tension you sat in without resolving."

Bad labels — avoid:
  "How are you feeling?" (vacuous)
  "Rate your day 1-10" (no angle)
  "What are you grateful for?" (wellness cliche)
  "List 3 things that went well" (productivity-bro)
  "Were you mindful?" (moralizing)
"""

_QGEN_SCHEMA_EXAMPLE = """{
  "questions": [
    {"id": "gen_edge", "type": "slider", "label": "How sharp was your edge today?", "weight": "light", "min": 0, "max": 10, "leftLabel": "dull", "rightLabel": "razor"},
    {"id": "gen_texture", "type": "tags", "label": "mark the texture of today", "weight": "medium", "options": ["smooth", "rough", "static", "splintered", "humid", "porous", "brittle"]},
    {"id": "gen_almost", "type": "shorttext", "label": "what did you almost not do today, but did?", "weight": "medium", "placeholder": "the one you nearly skipped..."},
    {"id": "gen_open_scene", "type": "longtext", "label": "describe today the way a film about you would open the scene.", "weight": "heavy", "placeholder": "// no rules. give the camera a place to land."}
  ],
  "meta": {"source": "gemini", "model_id": "gemini-2.5-flash", "latency_ms": 0}
}"""


def generate_questions(
    *,
    date_str: str,
    reroll_count: int = 0,
    accent: dict | None = None,
    recent_question_ids: list[str] | None = None,
    want: dict | None = None,
    vibe: str | None = None,
    history_oneliners: list[str] | None = None,
    max_tokens: int = 3072,
) -> dict | None:
    """Ask the LLM for fresh questions. Returns the response dict (questions + meta) or None on failure."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not _provider_key(provider):
        return None

    want_obj = want or {"light": 2, "medium": 2, "heavy": 1}
    light_n = max(0, min(6, int(want_obj.get("light", 2))))
    medium_n = max(0, min(6, int(want_obj.get("medium", 2))))
    heavy_n = max(0, min(3, int(want_obj.get("heavy", 1))))

    recent_ids = (recent_question_ids or [])[:60]
    history = (history_oneliners or [])[:30]

    user_prompt = (
        f"Generate fresh journal questions for {date_str}.\n\n"
        f"Want: light={light_n}, medium={medium_n}, heavy={heavy_n}\n"
        f"Reroll count today: {reroll_count} — higher means push harder for novelty.\n"
        f"Today's accent: {accent or {}}\n"
        f"Vibe nudge (free-text from user, may be empty): {vibe or ''}\n\n"
        f"Recent question ids to AVOID (the user has seen these): {recent_ids}\n"
        f"Recent oneliners (so you can riff/contrast without repeating): {history}\n\n"
        f"{_QGEN_GOOD_EXAMPLES}\n\n"
        f"Output ONLY a JSON object exactly matching this shape:\n{_QGEN_SCHEMA_EXAMPLE}\n\n"
        "Rules:\n"
        f"- Generate exactly {light_n + medium_n + heavy_n} items.\n"
        "- Each `id` must start with 'gen_' and contain only [a-z0-9_].\n"
        "- Each `type` must be one of: slider | choice | tags | shorttext | longtext | timeline.\n"
        "- Sliders need min/max/leftLabel/rightLabel. Choices/tags need options (3-10 items).\n"
        "- Heavies should be longtext or shorttext with provocative prompts.\n"
        "- Never invent moralizing questions. Never use wellness-app voice.\n"
        "- Don't include 'meta' — server sets it.\n"
    )

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=[{"role": "user", "content": user_prompt}],
        system=_QGEN_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="qgen",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    raw = _extract_text(response)
    parsed = _safe_parse_json(raw)
    if parsed is None:
        return None

    questions_in = parsed.get("questions")
    if not isinstance(questions_in, list):
        return None

    shaped_questions: list[dict] = []
    allowed_types = {"slider", "choice", "tags", "shorttext", "longtext", "timeline"}
    allowed_weights = {"light", "medium", "heavy"}

    import re as _re
    id_re = _re.compile(r"^gen_[a-z0-9_]{2,40}$")

    for q in questions_in[:12]:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id", "")).strip()
        if not id_re.fullmatch(qid):
            # Normalize: snake-case and prepend gen_ if missing.
            slug = _re.sub(r"[^a-z0-9_]", "_", qid.lower())
            slug = _re.sub(r"_+", "_", slug).strip("_")[:36]
            qid = f"gen_{slug}" if not slug.startswith("gen_") else slug
            if not id_re.fullmatch(qid):
                continue
        qtype = str(q.get("type", "")).strip().lower()
        if qtype not in allowed_types:
            continue
        weight = str(q.get("weight", "")).strip().lower()
        if weight not in allowed_weights:
            continue
        label = str(q.get("label", "")).strip()[:140]
        if not label:
            continue

        out: dict[str, Any] = {"id": qid, "type": qtype, "label": label, "weight": weight}

        if qtype == "slider":
            try:
                out["min"] = float(q.get("min", 0))
                out["max"] = float(q.get("max", 10))
            except (TypeError, ValueError):
                out["min"], out["max"] = 0, 10
            out["leftLabel"] = str(q.get("leftLabel", ""))[:60]
            out["rightLabel"] = str(q.get("rightLabel", ""))[:60]
        elif qtype in ("choice", "tags"):
            opts_in = q.get("options")
            if not isinstance(opts_in, list):
                continue
            opts_out = [str(o)[:60] for o in opts_in[:14] if isinstance(o, str) and o.strip()]
            if len(opts_out) < 2:
                continue
            out["options"] = opts_out
        elif qtype in ("shorttext", "longtext"):
            ph = q.get("placeholder")
            if isinstance(ph, str):
                out["placeholder"] = ph.strip()[:200]
        elif qtype == "timeline":
            segs_in = q.get("segments")
            segs_out: list[str] = []
            if isinstance(segs_in, list):
                segs_out = [str(s)[:30] for s in segs_in[:5] if isinstance(s, str) and s.strip()]
            if len(segs_out) < 2:
                segs_out = ["morning", "afternoon", "evening"]
            out["segments"] = segs_out

        shaped_questions.append(out)

    # Cap by weight bucket.
    by_weight: dict[str, list[dict]] = {"light": [], "medium": [], "heavy": []}
    for q in shaped_questions:
        by_weight[q["weight"]].append(q)
    final = by_weight["light"][:light_n] + by_weight["medium"][:medium_n] + by_weight["heavy"][:heavy_n]

    return {
        "questions": final,
        "meta": {
            "source": provider,
            "model_id": _resolve_model(provider),
            "latency_ms": latency_ms,
        },
    }


# ---- question enrichment --------------------------------------------------
# Different from generate_questions: enrichment takes EXISTING static QBANK
# items and refreshes their labels / placeholders / option lists to today's
# vibe + accent. Same ids, same types, fresh content. Useful at page load to
# make the static bank feel daily-novel.

_QENRICH_SYSTEM_PROMPT = (
    "You enrich existing journal questions for the writer. Given a list of "
    "questions with their static labels and option lists, return a SAME-LENGTH "
    "list where each item has been freshened to today's vibe — keeping the SAME "
    "id, SAME type, SAME weight, SAME min/max if slider — but with crisper "
    "labels, more vivid placeholder/option text, and more inspiring word choice. "
    "Never wellness-app voice. Never change the question's intent. Lowercase. "
    "Output ONLY valid JSON matching the schema. If a question is already "
    "perfect, keep it as-is (return the original)."
)

_QENRICH_SCHEMA_EXAMPLE = """{
  "questions": [
    {"id": "mood", "type": "slider", "label": "where did today land?", "weight": "light", "min": 0, "max": 10, "leftLabel": "underwater", "rightLabel": "rooftop"},
    {"id": "touched", "type": "tags", "label": "what surfaces did you spend yourself on?", "weight": "medium", "options": ["code", "people", "screens", "paper", "the body", "the city", "food", "no one"]}
  ],
  "meta": {"source": "gemini", "model_id": "gemini-2.5-flash", "latency_ms": 0}
}"""


def enrich_questions(
    *,
    date_str: str,
    seeds: list[dict],
    accent: dict | None = None,
    vibe: str | None = None,
    history_oneliners: list[str] | None = None,
    max_tokens: int = 4096,
) -> dict | None:
    """Take existing question objects, return enriched versions.

    Same ids / types / weights / min-max preserved. Only labels, placeholders,
    leftLabel/rightLabel, and options are freshened.
    """
    if not isinstance(seeds, list) or not seeds:
        return None

    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not _provider_key(provider):
        return None

    # Trim incoming seeds to what the LLM needs.
    seed_trimmed: list[dict] = []
    for s in seeds[:12]:
        if not isinstance(s, dict):
            continue
        t = {
            "id":     str(s.get("id", ""))[:40],
            "type":   str(s.get("type", ""))[:20],
            "label":  str(s.get("label", ""))[:140],
            "weight": str(s.get("weight", ""))[:20],
        }
        if not (t["id"] and t["type"] and t["weight"]):
            continue
        if "min" in s and "max" in s:
            try:
                t["min"] = float(s["min"])
                t["max"] = float(s["max"])
            except (TypeError, ValueError):
                pass
        for k in ("leftLabel", "rightLabel", "placeholder"):
            v = s.get(k)
            if isinstance(v, str):
                t[k] = v[:80]
        if isinstance(s.get("options"), list):
            t["options"] = [str(o)[:60] for o in s["options"][:14] if isinstance(o, str)]
        if isinstance(s.get("segments"), list):
            t["segments"] = [str(o)[:30] for o in s["segments"][:5] if isinstance(o, str)]
        seed_trimmed.append(t)

    if not seed_trimmed:
        return None

    user_prompt = (
        f"Date: {date_str}. Accent: {accent or {}}. Vibe nudge: {vibe or ''}.\n"
        f"Recent oneliners (do not echo): {(history_oneliners or [])[:14]}\n\n"
        f"Seed questions to enrich:\n{json.dumps(seed_trimmed, ensure_ascii=False)}\n\n"
        f"Output a JSON object exactly matching this shape (same length as input):\n"
        f"{_QENRICH_SCHEMA_EXAMPLE}\n\n"
        "Rules:\n"
        "- Return the SAME number of questions as the input, in the SAME order.\n"
        "- Preserve each question's id, type, and weight EXACTLY.\n"
        "- For sliders, preserve min/max EXACTLY; only update leftLabel/rightLabel/label.\n"
        "- For tags/choice, keep ~same number of options; freshen the words.\n"
        "- For shorttext/longtext, update label and placeholder.\n"
        "- For timeline, leave segments unless they're trivially generic.\n"
        "- Lowercase. Terse. Tech-tool voice. Never moralize. Never advise.\n"
    )

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=[{"role": "user", "content": user_prompt}],
        system=_QENRICH_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="qenrich",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _safe_parse_json(_extract_text(response))
    if parsed is None:
        return None
    questions_in = parsed.get("questions")
    if not isinstance(questions_in, list):
        return None

    # Match enriched outputs back to seeds by id; preserve ordering of seeds,
    # fall back to the seed itself if the LLM dropped an item or returned bad data.
    by_id: dict[str, dict] = {}
    for q in questions_in:
        if isinstance(q, dict) and isinstance(q.get("id"), str):
            by_id[q["id"]] = q

    out: list[dict] = []
    for seed in seeds[:12]:
        if not isinstance(seed, dict):
            continue
        sid = seed.get("id")
        enriched = by_id.get(sid) if isinstance(sid, str) else None
        if not enriched:
            out.append(seed)
            continue
        # Merge: keep seed's id/type/weight/min/max; take enriched's label/placeholder/options/etc.
        merged = {
            "id":     seed["id"],
            "type":   seed["type"],
            "weight": seed["weight"],
            "label":  str(enriched.get("label", seed.get("label", "")))[:140] or seed.get("label", ""),
        }
        if "min" in seed:
            merged["min"] = seed["min"]
        if "max" in seed:
            merged["max"] = seed["max"]
        for k in ("leftLabel", "rightLabel", "placeholder"):
            v = enriched.get(k, seed.get(k))
            if isinstance(v, str) and v.strip():
                merged[k] = v.strip()[:80]
        if "options" in seed:
            opts_in = enriched.get("options")
            if isinstance(opts_in, list) and len(opts_in) >= 2:
                merged["options"] = [str(o)[:60] for o in opts_in[:14] if isinstance(o, str) and o.strip()]
            else:
                merged["options"] = seed["options"]
        if "segments" in seed:
            segs_in = enriched.get("segments")
            if isinstance(segs_in, list) and len(segs_in) >= 2:
                merged["segments"] = [str(s)[:30] for s in segs_in[:5] if isinstance(s, str) and s.strip()]
            else:
                merged["segments"] = seed["segments"]
        out.append(merged)

    return {
        "questions": out,
        "meta": {
            "source": provider,
            "model_id": _resolve_model(provider),
            "latency_ms": latency_ms,
        },
    }


# ---- english analysis -----------------------------------------------------
# A four-section enriched read on the user's accumulated entries + feedback:
#   1. native_moves    — phrasebook of idiomatic patterns to internalize
#   2. drift_map       — recurring L2 patterns + counts
#   3. register_check  — per-entry register verdicts
#   4. vocabulary      — terms suggested over time, split into absorbed / pending
# Response conforms to .team/schemas/english.analysis.response.json.

_ENG_ANALYSIS_SYSTEM_PROMPT = (
    "You analyze accumulated journal data for the writer. Produce a structured "
    "four-section language analysis. Match the language in the supplied entries; "
    "if unclear, default to English. NEVER moralize. NEVER use generic wellness-app "
    "voice. Target polished language for all examples. Lowercase mono-feel for "
    "phrases; sans prose for notes. Output ONLY valid JSON matching the "
    "schema you're given.\n\n"
    "SECTION GOALS:\n"
    "- native_moves: a curated phrasebook of patterns the writer should add to their "
    "  active repertoire. Pull from his accumulated suggested-upgrades AND "
    "  suggest additional patterns that fit his register. "
    "  Each item is alive (not cliched). Example field uses HIS context.\n"
    "- drift_map: count recurring L2 patterns across the recent history "
    "  (preposition slips, l2 wordiness, register mixing, missed phrasal verbs, "
    "  hedge over-use, article slips, tense slips). Use actual counts you can "
    "  derive from the supplied data.\n"
    "- register_check: per recent entry — what register dominated and whether "
    "  it slipped. Caller passes a list of recent entry digests.\n"
    "- vocabulary: split into 'absorbed' (terms previously suggested AND now "
    "  used in user's own writing) vs 'pending' (suggested but not yet used). "
    "  Use accumulated feedback files as the source of truth."
)


def build_english_analysis(
    *,
    recent_entries: list[dict],
    recent_feedback: list[dict],
    window_days: int = 30,
    date_from: str | None = None,
    date_to:   str | None = None,
    max_tokens: int = 4096,
) -> dict | None:
    """Generate the four-section english analysis response.

    Args:
        recent_entries:  list of entry dicts (newest first), each conforms to
                         journal/entries/<date>.json shape.
        recent_feedback: list of feedback dicts (newest first), each conforms
                         to insight.response.json shape.
        window_days:     how many days the analysis should cover when no
                         explicit date range is given.
        date_from:       ISO YYYY-MM-DD; if set, filter entries to dates >=
                         date_from. Overrides window_days.
        date_to:         ISO YYYY-MM-DD; if set, filter entries to dates <=
                         date_to. Defaults to today when only date_from is set.

    Returns the response dict (4 sections + meta) or None on failure.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not _provider_key(provider):
        return None

    # If an explicit date range is supplied, filter both entries and feedback
    # to it; otherwise fall back to the window_days slice.
    if date_from:
        def _in_range(item: dict) -> bool:
            d = item.get("date")
            if not isinstance(d, str):
                return False
            if date_from and d < date_from:
                return False
            if date_to and d > date_to:
                return False
            return True
        recent_entries  = [e for e in recent_entries  if _in_range(e)]
        recent_feedback = [f for f in recent_feedback if _in_range(f)]
        # Compute effective window for the LLM prompt context.
        try:
            from datetime import date as _d
            d_from = _d.fromisoformat(date_from)
            d_to   = _d.fromisoformat(date_to) if date_to else _d.today()
            window_days = max(1, (d_to - d_from).days + 1)
        except Exception:
            pass

    # Trim payload to keep prompt size reasonable.
    entries_trimmed: list[dict] = []
    for e in recent_entries[:window_days]:
        if not isinstance(e, dict):
            continue
        item = {"date": e.get("date")}
        accent = e.get("accent")
        if isinstance(accent, dict):
            item["accent"] = {"name": accent.get("name")}
        # collect long-form text
        longtexts: list[str] = []
        oneliner = None
        for q in (e.get("questions") or []):
            if not isinstance(q, dict):
                continue
            v = q.get("value")
            if not isinstance(v, str):
                continue
            qid = q.get("id")
            if qid == "oneliner" and not oneliner:
                oneliner = v[:200]
            elif q.get("type") in ("longtext", "shorttext"):
                longtexts.append(v[:600])
        if oneliner:
            item["oneliner"] = oneliner
        if longtexts:
            item["text"] = " | ".join(longtexts)[:1800]
        entries_trimmed.append(item)

    feedback_trimmed: list[dict] = []
    for f in recent_feedback[:window_days]:
        if not isinstance(f, dict):
            continue
        item: dict[str, Any] = {}
        eng = f.get("english")
        if isinstance(eng, list):
            item["english"] = [
                {
                    "original": str(e.get("original", ""))[:200],
                    "suggested": str(e.get("suggested", ""))[:200],
                    "note": str(e.get("note", ""))[:240],
                    "kind": str(e.get("kind", ""))[:32],
                }
                for e in eng[:5] if isinstance(e, dict)
            ]
        ref = f.get("reflect")
        if isinstance(ref, list):
            item["reflect_kinds"] = [
                str(r.get("kind", "other"))[:24]
                for r in ref[:5] if isinstance(r, dict)
            ]
        feedback_trimmed.append(item)

    user_prompt = (
        f"window: last {window_days} days\n"
        f"recent entries (newest first):\n{json.dumps(entries_trimmed, ensure_ascii=False)}\n\n"
        f"recent feedback (newest first):\n{json.dumps(feedback_trimmed, ensure_ascii=False)}\n\n"
        "Output a JSON object with four sections + meta:\n"
        "  native_moves: { phrasal_verbs: [...], preposition_pairs: [...], "
        "register_moves: [...], idioms: [...] } (each capped at 6)\n"
        "  drift_map: { window_days: N, patterns: [{pattern, count, note}] } "
        "(up to 8)\n"
        "  register_check: { entries: [{date, dominant, verdict, slips?}] } "
        "(up to 10, newest first; dominant in {casual, business, academic, mixed, neutral})\n"
        "  vocabulary: { absorbed: [{term, first_seen?, used_count?, "
        "context_note?}], pending: [{term, suggested_at?, context_note?}] } "
        "(each up to 30)\n"
        "  meta: leave empty — server fills it.\n\n"
        "Rules:\n"
        "- Target polished language throughout. Skip basic upgrades.\n"
        "- For vocabulary.absorbed: only include terms that appear in BOTH "
        "the suggested-upgrades AND the user's own subsequent writing.\n"
        "- For native_moves: aim for vivid + currently-used idiomatic patterns. Avoid clichés.\n"
        "- For drift_map: count from the actual accumulated kinds in recent_feedback.\n"
        "- For register_check: read each entry's text and judge.\n"
        "- Lowercase. Tech-tool voice. No moralizing."
    )

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=[{"role": "user", "content": user_prompt}],
        system=_ENG_ANALYSIS_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="eng-analysis",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _safe_parse_json(_extract_text(response))
    if parsed is None:
        return None

    out: dict[str, Any] = {
        "native_moves": _shape_native_moves(parsed.get("native_moves")),
        "drift_map":    _shape_drift_map(parsed.get("drift_map"), window_days),
        "register_check": _shape_register_check(parsed.get("register_check")),
        "vocabulary":   _shape_vocabulary(parsed.get("vocabulary")),
        "meta": {
            "source":     provider,
            "model_id":   _resolve_model(provider),
            "latency_ms": latency_ms,
        },
    }
    return out


def _shape_native_moves(d: Any) -> dict:
    out: dict[str, list] = {
        "phrasal_verbs": [], "preposition_pairs": [],
        "register_moves": [], "idioms": [],
    }
    if not isinstance(d, dict):
        return out
    for item in (d.get("phrasal_verbs") or [])[:6]:
        if isinstance(item, dict) and isinstance(item.get("form"), str) and isinstance(item.get("example"), str):
            o = {"form": item["form"][:80], "example": item["example"][:200]}
            if isinstance(item.get("when"), str):
                o["when"] = item["when"][:160]
            out["phrasal_verbs"].append(o)
    for item in (d.get("preposition_pairs") or [])[:6]:
        if isinstance(item, dict) and isinstance(item.get("pair"), str) and isinstance(item.get("note"), str):
            out["preposition_pairs"].append({
                "pair": item["pair"][:80], "note": item["note"][:200],
            })
    for item in (d.get("register_moves") or [])[:6]:
        if (isinstance(item, dict) and isinstance(item.get("from"), str)
                and isinstance(item.get("to"), str) and isinstance(item.get("note"), str)):
            out["register_moves"].append({
                "from": item["from"][:120], "to": item["to"][:120],
                "note": item["note"][:200],
            })
    for item in (d.get("idioms") or [])[:6]:
        if isinstance(item, dict) and isinstance(item.get("idiom"), str) and isinstance(item.get("gloss"), str):
            out["idioms"].append({"idiom": item["idiom"][:80], "gloss": item["gloss"][:200]})
    return out


def _shape_drift_map(d: Any, window_days: int) -> dict:
    patterns: list[dict] = []
    if isinstance(d, dict):
        for item in (d.get("patterns") or [])[:8]:
            if not isinstance(item, dict):
                continue
            pat = item.get("pattern")
            note = item.get("note", "")
            try:
                count = int(item.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            if isinstance(pat, str) and pat.strip():
                patterns.append({
                    "pattern": pat.strip()[:80],
                    "count":   max(0, count),
                    "note":    str(note)[:300],
                })
    return {"window_days": window_days, "patterns": patterns}


def _shape_register_check(d: Any) -> dict:
    entries: list[dict] = []
    allowed = {"casual", "business", "academic", "mixed", "neutral"}
    if isinstance(d, dict):
        for item in (d.get("entries") or [])[:10]:
            if not isinstance(item, dict):
                continue
            date_str = item.get("date")
            if not (isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == "-"):
                continue
            dominant = str(item.get("dominant", "")).strip().lower()
            if dominant not in allowed:
                dominant = "neutral"
            verdict = str(item.get("verdict", "")).strip()[:240]
            row = {"date": date_str, "dominant": dominant, "verdict": verdict}
            slips_in = item.get("slips")
            if isinstance(slips_in, list):
                slips_out = []
                for s in slips_in[:4]:
                    if isinstance(s, dict) and isinstance(s.get("phrase"), str) and isinstance(s.get("issue"), str):
                        slips_out.append({"phrase": s["phrase"][:160], "issue": s["issue"][:200]})
                if slips_out:
                    row["slips"] = slips_out
            entries.append(row)
    return {"entries": entries}


def _shape_vocabulary(d: Any) -> dict:
    absorbed: list[dict] = []
    pending: list[dict] = []
    if isinstance(d, dict):
        for item in (d.get("absorbed") or [])[:30]:
            if not isinstance(item, dict):
                continue
            term = item.get("term")
            if not (isinstance(term, str) and term.strip()):
                continue
            o = {"term": term.strip()[:80]}
            if isinstance(item.get("first_seen"), str):
                o["first_seen"] = item["first_seen"][:10]
            try:
                if "used_count" in item:
                    o["used_count"] = max(1, int(item["used_count"]))
            except (TypeError, ValueError):
                pass
            if isinstance(item.get("context_note"), str):
                o["context_note"] = item["context_note"][:200]
            absorbed.append(o)
        for item in (d.get("pending") or [])[:30]:
            if not isinstance(item, dict):
                continue
            term = item.get("term")
            if not (isinstance(term, str) and term.strip()):
                continue
            o = {"term": term.strip()[:80]}
            if isinstance(item.get("suggested_at"), str):
                o["suggested_at"] = item["suggested_at"][:10]
            if isinstance(item.get("context_note"), str):
                o["context_note"] = item["context_note"][:200]
            pending.append(o)
    return {"absorbed": absorbed, "pending": pending}




# ─── track summary ───────────────────────────────────────────────────────
_TRACK_SUMMARY_PROMPT = (
    "You produce a brief, honest summary of someone's recent journaling for the "
    "track view of their journal platform. Read all the entries supplied. Output ONLY "
    "valid JSON matching the schema you're given. No prose around the JSON.\n\n"
    "RULES:\n"
    "- summary: 2-3 short sentences. Surface the underlying themes, NOT a recap. "
    "Tone: dry, observational, the way a perceptive friend would describe what "
    "they noticed. Lowercase. No moralizing. No coaching.\n"
    "- shifts: 1-3 short observations of what's notably DIFFERENT in the recent "
    "entries vs earlier ones in the window. If the window is too short for shifts, "
    "return an empty array.\n"
    "- memorable_lines: 3-6 quotes lifted verbatim from the entries. Each must be "
    "ACTUALLY present in the supplied text (don't invent). Prefer lines that are "
    "specific, vivid, or self-revealing. Format: {date, quote}.\n"
    "- momentum: { trend: 'rising'|'steady'|'dipping'|'sparse'|'new', note: short "
    "one-liner }. Base on word count and frequency.\n"
    "- Keep all strings lowercase unless quoting proper nouns from entries verbatim."
)

def build_track_summary(
    *,
    recent_entries: list,
    window_days: int = 30,
    max_tokens: int = 1500,
) -> dict | None:
    """Generate the track view summary from accumulated entries.

    Returns a dict matching track.summary.response.json or None on failure.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if not _provider_key(provider):
        return None

    # Build a digest of entries: date + word count + raw text snippet.
    entries_digest = []
    total_words = 0
    for e in (recent_entries or [])[:window_days]:
        if not isinstance(e, dict):
            continue
        date = e.get("date")
        if not isinstance(date, str):
            continue
        texts = []
        for q in (e.get("questions") or []):
            if not isinstance(q, dict):
                continue
            v = q.get("value")
            if isinstance(v, str) and q.get("type") in ("longtext", "shorttext"):
                texts.append(v[:500])
        text_blob = " | ".join(texts)
        words = len(text_blob.split())
        total_words += words
        if text_blob.strip():
            entries_digest.append({"date": date, "words": words, "text": text_blob[:1200]})

    if not entries_digest:
        return None

    user_prompt = (
        f"Window: last {window_days} days.\n"
        f"Total entries with text: {len(entries_digest)}.\n"
        f"Total words across window: {total_words}.\n\n"
        f"Entries (newest first):\n{json.dumps(entries_digest, ensure_ascii=False)}\n\n"
        "Output a JSON object exactly matching: { summary, shifts[], "
        "memorable_lines[{date,quote}], momentum{trend,note}, meta }.\n"
        "- memorable_lines: 3-6 items, each quote MUST appear verbatim in the entries above.\n"
        "- shifts: 0-3 items.\n"
        "- summary: 2-3 short sentences. Lowercase."
    )

    t0 = time.perf_counter()
    response = _chat_with_retry(
        messages=[{"role": "user", "content": user_prompt}],
        system=_TRACK_SUMMARY_PROMPT,
        max_tokens=max_tokens,
        model=_resolve_model(provider),
        label="track-summary",
    )
    if response is None:
        return None
    latency_ms = int((time.perf_counter() - t0) * 1000)

    parsed = _safe_parse_json(_extract_text(response))
    if parsed is None:
        return None

    # Shape defensively
    out = {
        "summary": str(parsed.get("summary",""))[:600],
        "shifts": [str(s)[:200] for s in (parsed.get("shifts") or []) if isinstance(s, str)][:4],
        "memorable_lines": [],
        "momentum": {
            "trend": parsed.get("momentum",{}).get("trend") if isinstance(parsed.get("momentum"), dict) else None,
            "note":  parsed.get("momentum",{}).get("note") if isinstance(parsed.get("momentum"), dict) else None,
        },
        "meta": {
            "source": provider,
            "model_id": _resolve_model(provider),
            "latency_ms": latency_ms,
            "window_days": window_days,
        },
    }
    if out["momentum"]["trend"] not in ("rising","steady","dipping","sparse","new"):
        out["momentum"]["trend"] = "steady"
    if not out["momentum"]["note"]:
        out["momentum"]["note"] = ""
    for ml in (parsed.get("memorable_lines") or [])[:6]:
        if isinstance(ml, dict) and isinstance(ml.get("date"), str) and isinstance(ml.get("quote"), str):
            out["memorable_lines"].append({
                "date": ml["date"][:10],
                "quote": ml["quote"].strip()[:280],
            })
    return out


# ---- standalone test entry point -----------------------------------------

if __name__ == "__main__":
    # Quick smoke test — run with: python insight.py
    sample_entry = {
        "date": "2026-05-13",
        "weekday": "Tuesday",
        "accent": {"name": "acid", "hex": "#b5fa3a"},
        "saved_at": "2026-05-13T14:32:08Z",
        "questions": [
            {
                "id": "oneliner",
                "type": "shorttext",
                "label": "Sum up today",
                "value": "I was really tired today and could not focus, but I should push through anyway.",
            },
            {
                "id": "free",
                "type": "longtext",
                "label": "Anywhere your mind wants to go",
                "value": "Long day. I am supposed to be working on the model but I keep thinking about whether I should switch to PE. Maybe I should. I should probably just commit. I do not want to start over.",
            },
        ],
    }
    sample_history: list[dict] = []
    print("provider:", os.environ.get("LLM_PROVIDER", "gemini"))
    print("model:", _resolve_model(os.environ.get("LLM_PROVIDER", "gemini").lower()))
    result = build_insight(sample_entry, sample_history)
    if result is None:
        print("FAIL — no key set, or call failed. See stderr.")
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
