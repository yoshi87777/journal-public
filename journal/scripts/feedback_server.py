"""life-os local feedback server (PLAN.md §2.2 Option A).

A tiny FastAPI app that runs on http://127.0.0.1:5757 and gives the
journal page a "smart" feedback path.

Design notes:
- Binds 127.0.0.1 ONLY. Never 0.0.0.0.
- Validates Host header (defends against DNS rebinding).
- Rejects request bodies > 100KB at the middleware layer.
- Date strings are validated with re.fullmatch(^\\d{4}-\\d{2}-\\d{2}$).
- All filesystem writes are confined to JOURNAL_DIR/{entries,feedback}.
- The frontend is opened as a local file://, so CORS allows origin "null"
  and http://localhost:* / http://127.0.0.1:*. LIFEOS_DEV=1 widens this
  to "*" for development only — startup logs a loud warning.
- ANTHROPIC_API_KEY is read from env. It is NEVER logged or echoed.
- Entry content is NEVER logged. Only method/path/status/latency.

Run:
    python feedback_server.py
or via:
    bash start.sh
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import feedback_rules
import insight as insight_engine

# ---- paths -----------------------------------------------------------------

JOURNAL_DIR = Path(__file__).resolve().parent.parent
ENTRIES_DIR = JOURNAL_DIR / "entries"
FEEDBACK_DIR = JOURNAL_DIR / "feedback"
CONVERSATIONS_DIR = JOURNAL_DIR / "conversations"
ENTRIES_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---- constants -------------------------------------------------------------

# Try to load the project .env before anything else reads env vars.
def _bootstrap_env() -> None:
    """Load only this project's .env.

    override=True means the .env value beats any pre-existing shell var,
    so editing journal-share/.env and restarting always takes effect (no stale
    shell-export surprise).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent.parent.parent
    env_path = here / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        provider = os.environ.get("LLM_PROVIDER", "(default: gemini)")
        model = os.environ.get("MODEL_ID", "(default per provider)")
        print(
            f"[journal] env loaded from {env_path} · LLM_PROVIDER={provider} · MODEL_ID={model}",
            file=sys.stderr,
        )


_bootstrap_env()


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAX_BODY_BYTES = 100 * 1024  # 100KB hard cap

# Host allowlist for the DNS-rebinding defense. Local defaults + any extra
# hosts from the LIFEOS_ALLOWED_HOSTS env var (comma-separated). Production
# deployments add e.g. "aria-anova.xyz" via /etc/life-os/env.
_DEFAULT_HOSTS = {"localhost", "127.0.0.1"}
_extra_hosts = os.environ.get("LIFEOS_ALLOWED_HOSTS", "")
ALLOWED_HOSTS = _DEFAULT_HOSTS | {
    h.strip().lower() for h in _extra_hosts.split(",") if h.strip()
}

LIFEOS_DEV = os.environ.get("LIFEOS_DEV") == "1"

_PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "google":    "GEMINI_API_KEY",
}


def _engine_name() -> str:
    """Return the active engine: <provider> if its key is set, else 'rules'."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    key_var = _PROVIDER_KEY_VARS.get(provider, "GEMINI_API_KEY")
    if os.environ.get(key_var) or os.environ.get("GOOGLE_API_KEY"):
        return provider
    return "rules"


# ---- safe path helpers -----------------------------------------------------

def _validated_date(s: str) -> str:
    if not isinstance(s, str) or not DATE_RE.fullmatch(s):
        raise HTTPException(status_code=400, detail="invalid date format; expected YYYY-MM-DD")
    return s


def _safe_path(base: Path, date_str: str, suffix: str = ".json") -> Path:
    """Return base/<date_str><suffix>, validated to live inside base."""
    _validated_date(date_str)
    candidate = (base / f"{date_str}{suffix}").resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid path")
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="invalid path")
    return candidate


# ---- middleware ------------------------------------------------------------

class BodySizeAndHostMiddleware(BaseHTTPMiddleware):
    """Enforce Content-Length cap and Host header allowlist before any handler runs."""

    async def dispatch(self, request: Request, call_next):
        # Host header check (DNS-rebinding defense).
        host = (request.headers.get("host") or "").split(":")[0].lower()
        if host and host not in ALLOWED_HOSTS:
            return JSONResponse(status_code=400, content={"error": "invalid host"})

        # Body size check, BEFORE we read the body into memory.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_BODY_BYTES:
                    return JSONResponse(status_code=413, content={"error": "request body too large"})
            except ValueError:
                return JSONResponse(status_code=400, content={"error": "invalid Content-Length"})
        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """One log line per request. Never logs body/query content."""

    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            ts = datetime.now().strftime("%H:%M:%S")
            # No exception details; the global handler will produce the response.
            print(f"[{ts}] {request.method} {request.url.path} -> 500 ({elapsed:.0f}ms)", file=sys.stderr)
            raise
        elapsed = (time.perf_counter() - t0) * 1000
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {request.method} {request.url.path} -> {status} ({elapsed:.0f}ms)", file=sys.stderr)
        return response


# ---- app -------------------------------------------------------------------

app = FastAPI(title="life-os feedback server", docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(AccessLogMiddleware)
app.add_middleware(BodySizeAndHostMiddleware)

if LIFEOS_DEV:
    print("[life-os] WARNING: LIFEOS_DEV=1 — CORS open to all origins", file=sys.stderr)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        # "null" is the Origin sent by browsers for file:// pages.
        allow_origins=["null"],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )


@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak tracebacks or env to the client.
    return JSONResponse(status_code=500, content={"error": "internal server error"})


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


# ---- history loader --------------------------------------------------------

def _load_history(exclude_date: str | None = None, limit: int = 60) -> list[dict]:
    """Load past entries (newest first), optionally excluding one date."""
    items: list[tuple[str, dict]] = []
    for p in sorted(ENTRIES_DIR.glob("*.json"), reverse=True):
        name = p.stem
        if not DATE_RE.fullmatch(name):
            continue
        if exclude_date and name == exclude_date:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        items.append((name, data))
        if len(items) >= limit:
            break
    return [d for _, d in items]


# ---- insight (provider-agnostic LLM call) ---------------------------------
# All prompt + schema lives in journal/scripts/insight.py — same module is
# used by .team/run.py role=insight. Provider is selected by LLM_PROVIDER
# (anthropic | openai | gemini).


# ---- routes ----------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"ok": True, "engine": _engine_name()}


@app.post("/api/journal")
async def post_journal(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    entry = payload.get("entry")
    if not isinstance(entry, dict):
        raise HTTPException(status_code=400, detail="missing 'entry' object")

    date_str = entry.get("date")
    if not isinstance(date_str, str) or not DATE_RE.fullmatch(date_str):
        raise HTTPException(status_code=400, detail="entry.date must be YYYY-MM-DD")

    entry_path = _safe_path(ENTRIES_DIR, date_str)
    feedback_path = _safe_path(FEEDBACK_DIR, date_str)

    new_bytes = json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8")

    # If file exists and differs, back it up with a timestamp suffix.
    if entry_path.exists():
        try:
            existing = entry_path.read_bytes()
        except OSError:
            existing = b""
        if existing and existing != new_bytes:
            ts = datetime.now().strftime("%H%M%S")
            backup = entry_path.with_suffix(entry_path.suffix + f".{ts}.bak")
            try:
                backup.write_bytes(existing)
            except OSError:
                pass

    try:
        entry_path.write_bytes(new_bytes)
    except OSError:
        raise HTTPException(status_code=500, detail="failed to write entry")

    history = _load_history(exclude_date=date_str)

    t0 = time.perf_counter()
    llm_payload = insight_engine.build_insight(entry, history)
    rules_feedback = feedback_rules.build_feedback(entry, history)

    if llm_payload is not None:
        # Merge: LLM owns english + reflect + patterns + prompt; rules own gamification.
        feedback = {
            "english": llm_payload.get("english") or [],
            "reflect": llm_payload.get("reflect") or [],
            "gamification": rules_feedback["gamification"],
            "patterns": llm_payload.get("patterns") or rules_feedback["patterns"],
            "prompt_for_tomorrow": llm_payload.get("prompt_for_tomorrow") or rules_feedback["prompt_for_tomorrow"],
            "meta": {
                "source": llm_payload.get("meta", {}).get("source", _engine_name()),
                "model_id": llm_payload.get("meta", {}).get("model_id"),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            },
        }
    else:
        feedback = rules_feedback
        feedback.setdefault("reflect", [])
        feedback["meta"]["latency_ms"] = int((time.perf_counter() - t0) * 1000)

    # Optional badge: new streak milestones.
    streak = feedback["gamification"].get("streak", 0)
    if streak in {3, 5, 7, 14, 30, 60, 100, 365}:
        feedback["gamification"]["new_badge"] = f"first {streak}-day streak"

    try:
        feedback_path.write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Persisting feedback failed; still return it to the client.
        pass

    return feedback


@app.get("/api/journal/feedback/{date_str}")
async def get_feedback(date_str: str) -> Any:
    _validated_date(date_str)
    path = _safe_path(FEEDBACK_DIR, date_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail="feedback not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="failed to read feedback")
    return data


# ---- conversation endpoints ----------------------------------------------
# Feedback can spark a conversation. The user clicks "reply" on a reflect or
# english item, types a response. The server appends to a per-day thread, asks
# the insight engine for a short follow-up, appends that, and returns the
# whole thread.

def _new_turn_id() -> str:
    return f"t-{int(time.time() * 1000)}-{os.urandom(2).hex()}"


def _load_thread(date_str: str) -> dict:
    path = _safe_path(CONVERSATIONS_DIR, date_str)
    if not path.exists():
        return {"date": date_str, "turns": [], "summary": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("turns"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"date": date_str, "turns": [], "summary": None}


def _save_thread(date_str: str, thread: dict) -> None:
    path = _safe_path(CONVERSATIONS_DIR, date_str)
    path.write_text(json.dumps(thread, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/journal/conversation/{date_str}")
async def get_conversation(date_str: str) -> dict:
    _validated_date(date_str)
    return _load_thread(date_str)


@app.post("/api/journal/questions/generate")
async def post_questions_generate(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    date_str = payload.get("date")
    if not isinstance(date_str, str) or not DATE_RE.fullmatch(date_str):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    reroll_count = payload.get("reroll_count", 0)
    try:
        reroll_count = max(0, int(reroll_count))
    except (TypeError, ValueError):
        reroll_count = 0

    accent = payload.get("accent") if isinstance(payload.get("accent"), dict) else None
    recent_ids = payload.get("recent_question_ids")
    if not isinstance(recent_ids, list):
        recent_ids = []
    want = payload.get("want") if isinstance(payload.get("want"), dict) else None
    vibe = payload.get("vibe") if isinstance(payload.get("vibe"), str) else None
    history_oneliners = payload.get("history_oneliners")
    if not isinstance(history_oneliners, list):
        # Pull oneliners from existing entries as a server-side fallback.
        history_oneliners = []
        for h in _load_history(limit=14):
            qs = h.get("questions") or []
            if not isinstance(qs, list):
                continue
            for q in qs:
                if isinstance(q, dict) and q.get("id") == "oneliner" and isinstance(q.get("value"), str):
                    history_oneliners.append(q["value"][:160])
                    break

    result = insight_engine.generate_questions(
        date_str=date_str,
        reroll_count=reroll_count,
        accent=accent,
        recent_question_ids=[str(i) for i in recent_ids][:60],
        want=want,
        vibe=vibe,
        history_oneliners=history_oneliners,
    )

    if result is None:
        # Could be: no provider key, transient model overload, or schema mismatch.
        # All three end in the same place — caller falls back to static QBANK.
        raise HTTPException(status_code=503, detail="question generation temporarily unavailable")

    return result


@app.post("/api/journal/questions/enrich")
async def post_questions_enrich(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    date_str = payload.get("date")
    if not isinstance(date_str, str) or not DATE_RE.fullmatch(date_str):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise HTTPException(status_code=400, detail="seeds (list of question objects) required")

    accent = payload.get("accent") if isinstance(payload.get("accent"), dict) else None
    vibe = payload.get("vibe") if isinstance(payload.get("vibe"), str) else None
    history_oneliners = payload.get("history_oneliners")
    if not isinstance(history_oneliners, list):
        history_oneliners = []

    result = insight_engine.enrich_questions(
        date_str=date_str,
        seeds=seeds,
        accent=accent,
        vibe=vibe,
        history_oneliners=[str(h)[:200] for h in history_oneliners][:30],
    )

    if result is None:
        raise HTTPException(status_code=503, detail="question enrichment temporarily unavailable")

    return result


@app.post("/api/journal/track/summary")
async def post_track_summary(request: Request) -> dict:
    """LLM summary for the track overlay: themes, shifts, memorable lines, momentum."""
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        payload = {}
    try:
        window_days = max(1, min(int(payload.get("window_days", 30)), 365))
    except (TypeError, ValueError):
        window_days = 30
    recent_entries = _load_history(limit=window_days)
    result = insight_engine.build_track_summary(
        recent_entries=recent_entries,
        window_days=window_days,
    )
    if result is None:
        raise HTTPException(status_code=503, detail="track summary temporarily unavailable")
    return result


@app.post("/api/english/analysis")
async def post_english_analysis(request: Request) -> dict:
    """Generate the four-section english analysis (native_moves / drift_map /
    register_check / vocabulary) from the user's recent entries + feedback.
    """
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    try:
        window_days = max(1, min(int(payload.get("window_days", 30)), 365))
    except (TypeError, ValueError):
        window_days = 30

    # Optional explicit date range. ISO YYYY-MM-DD. If supplied, overrides
    # window_days. `mode` shortcuts: "week"/"month"/"year" / "all".
    date_from = payload.get("date_from") if isinstance(payload.get("date_from"), str) else None
    date_to   = payload.get("date_to")   if isinstance(payload.get("date_to"),   str) else None
    mode      = payload.get("mode")      if isinstance(payload.get("mode"),      str) else None
    if mode and not date_from:
        from datetime import date as _d, timedelta as _td
        today_d = _d.today()
        days = {"day": 1, "week": 7, "month": 30, "quarter": 90, "year": 365, "all": 3650}.get(mode)
        if days:
            date_from = (today_d - _td(days=days - 1)).isoformat()
            date_to   = today_d.isoformat()

    # Validate the ISO date strings (defense against bad input).
    if date_from and not DATE_RE.fullmatch(date_from):
        raise HTTPException(status_code=400, detail="date_from must be YYYY-MM-DD")
    if date_to and not DATE_RE.fullmatch(date_to):
        raise HTTPException(status_code=400, detail="date_to must be YYYY-MM-DD")

    # Load enough history. If a date range is set, load wider to filter inside the engine.
    load_limit = window_days
    if date_from:
        try:
            from datetime import date as _d
            d_from = _d.fromisoformat(date_from)
            d_to   = _d.fromisoformat(date_to) if date_to else _d.today()
            load_limit = max(window_days, (d_to - d_from).days + 1)
        except Exception:
            pass

    recent_entries = _load_history(limit=load_limit)

    recent_feedback: list[dict] = []
    for p in sorted(FEEDBACK_DIR.glob("*.json"), reverse=True):
        name = p.stem
        if not DATE_RE.fullmatch(name):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            # Insight.response.json doesn't carry the date; tag it from the filename
            # so build_english_analysis can filter by date.
            data.setdefault("date", name)
            recent_feedback.append(data)
        if len(recent_feedback) >= load_limit:
            break

    result = insight_engine.build_english_analysis(
        recent_entries=recent_entries,
        recent_feedback=recent_feedback,
        window_days=window_days,
        date_from=date_from,
        date_to=date_to,
    )

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="english analysis temporarily unavailable",
        )

    return result


@app.post("/api/journal/reply")
async def post_reply(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="request body too large")
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    date_str = payload.get("date")
    if not isinstance(date_str, str) or not DATE_RE.fullmatch(date_str):
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")

    user_text = payload.get("user_text")
    if not isinstance(user_text, str) or not user_text.strip():
        raise HTTPException(status_code=400, detail="user_text required")
    user_text = user_text.strip()[:4000]

    anchor = payload.get("anchor") if isinstance(payload.get("anchor"), dict) else None
    in_reply_to = payload.get("in_reply_to") if isinstance(payload.get("in_reply_to"), str) else None

    # Load existing thread and entry + insight snapshot for context.
    thread = _load_thread(date_str)
    entry_path = _safe_path(ENTRIES_DIR, date_str)
    feedback_path = _safe_path(FEEDBACK_DIR, date_str)
    entry: dict | None = None
    insight_snapshot: dict | None = None
    if entry_path.exists():
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entry = None
    if feedback_path.exists():
        try:
            insight_snapshot = json.loads(feedback_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            insight_snapshot = None

    iso_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    user_turn = {
        "turn_id": _new_turn_id(),
        "role": "user",
        "text": user_text,
        "created_at": iso_now,
        "in_reply_to": in_reply_to,
        "anchor": anchor,
    }
    thread["turns"].append(user_turn)

    # Build the assistant reply.
    reply = insight_engine.build_conversation_reply(
        entry=entry,
        insight_snapshot=insight_snapshot,
        anchor=anchor,
        turns=thread["turns"],
        user_text=user_text,
    )

    if reply is None:
        # No LLM available — produce a minimal acknowledgement so the UI has
        # something to render.
        assistant_text = (
            "noted. (no llm reachable — set LLM_PROVIDER + key to get a real reply.)"
        )
        meta = {"source": "rules", "model_id": None, "latency_ms": 0}
    else:
        assistant_text = reply["text"]
        meta = reply.get("meta") or {"source": _engine_name(), "model_id": None, "latency_ms": 0}

    assistant_turn = {
        "turn_id": _new_turn_id(),
        "role": "assistant",
        "text": assistant_text,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "in_reply_to": user_turn["turn_id"],
        "anchor": anchor,
        "meta": meta,
    }
    thread["turns"].append(assistant_turn)

    try:
        _save_thread(date_str, thread)
    except OSError:
        pass

    return {"assistant_turn": assistant_turn, "thread": thread}


@app.get("/api/journal/today")
async def get_today() -> Any:
    today_str = date.today().strftime("%Y-%m-%d")
    path = _safe_path(ENTRIES_DIR, today_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="failed to read today's entry")


@app.get("/api/journal/archive")
async def get_archive(limit: int = 30) -> list[dict]:
    try:
        n = int(limit)
    except (TypeError, ValueError):
        n = 30
    n = max(1, min(n, 365))

    out: list[dict] = []
    for p in sorted(ENTRIES_DIR.glob("*.json"), reverse=True):
        name = p.stem
        if not DATE_RE.fullmatch(name):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        data.setdefault("date", name)
        feedback_path = _safe_path(FEEDBACK_DIR, name)
        if feedback_path.exists():
            try:
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                feedback = None
            if isinstance(feedback, dict):
                english = feedback.get("english")
                if isinstance(english, list):
                    data["english_notes"] = english

        out.append(data)
        if len(out) >= n:
            break
    return out


# ---- entrypoint ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    engine = _engine_name()
    print(
        f"[life-os] feedback server :: engine={engine} :: bound to http://127.0.0.1:5757",
        file=sys.stderr,
    )
    if LIFEOS_DEV:
        print("[life-os] dev mode active (CORS=*) — DO NOT expose this port", file=sys.stderr)

    uvicorn.run(app, host="127.0.0.1", port=5757, log_level="warning")
