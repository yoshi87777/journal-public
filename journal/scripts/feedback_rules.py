"""Rule-based feedback fallback for the life-os journal.

Used when ANTHROPIC_API_KEY is not set, or when the Anthropic call fails.
Pure stdlib. Designed so the user still gets something cute and gamified.

Contract:
    build_feedback(entry: dict, history: list[dict]) -> dict
        Returns the feedback JSON shape documented in PLAN.md §2.2.
        meta.source is always "rules" here; latency is filled in by the caller.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any


# ---- english upgrades ------------------------------------------------------

# "very <adj>" → punchier single word.
_VERY_SWAPS: dict[str, str] = {
    "tired": "wiped",
    "sleepy": "wiped",
    "exhausted": "wiped",
    "good": "solid",
    "nice": "solid",
    "great": "stellar",
    "bad": "rough",
    "hard": "brutal",
    "difficult": "brutal",
    "happy": "thrilled",
    "sad": "down",
    "angry": "pissed",
    "busy": "slammed",
    "hungry": "starving",
    "cold": "freezing",
    "hot": "scorching",
    "scary": "creepy",
    "boring": "dull",
    "fast": "blazing",
    "slow": "sluggish",
    "interesting": "fascinating",
    "important": "critical",
    "small": "tiny",
    "big": "huge",
    "smart": "sharp",
    "stupid": "dumb",
}

_VERY_NOTE = "Native casual writing usually picks a stronger word over 'very + adj'."

# Negation → contraction.
_CONTRACTIONS: list[tuple[str, str]] = [
    ("could not", "couldn't"),
    ("would not", "wouldn't"),
    ("should not", "shouldn't"),
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("has not", "hasn't"),
    ("have not", "haven't"),
    ("had not", "hadn't"),
    ("will not", "won't"),
    ("cannot", "can't"),
    ("can not", "can't"),
]

_CONTRACTION_NOTE = "Casual writing contracts these."

_TOMORROW_PROMPTS: list[str] = [
    "Tomorrow: try writing the heavy entry in 3 sentences only.",
    "Tomorrow: pick the slider you'd most want to be wrong about.",
    "Tomorrow: oneliner in past-tense only.",
    "Tomorrow: leave one field blank on purpose.",
    "Tomorrow: write to your future self, 1 year out.",
    "Tomorrow: name today's color in 3 words, not hex.",
    "Tomorrow: timeline-only entry — no longform.",
    "Tomorrow: an observation of someone, not yourself.",
]

# Words we strip when scanning for "patterns" — too generic to flag.
_STOPWORDS: set[str] = {
    "a", "an", "and", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "as", "but",
    "or", "if", "then", "so", "not", "no", "do", "did", "does", "have", "has",
    "had", "i", "im", "ive", "id", "ill", "you", "your", "youre", "he", "she",
    "it", "we", "they", "them", "us", "my", "me", "his", "her", "their", "our",
    "this", "that", "these", "those", "there", "here", "what", "when", "where",
    "why", "how", "who", "which", "while", "about", "just", "really", "very",
    "more", "some", "any", "all", "every", "today", "tomorrow", "yesterday",
    "day", "week", "time", "now", "still", "even", "also", "too", "much",
    "get", "got", "make", "made", "go", "going", "went", "come", "came",
    "can", "could", "would", "should", "will", "may", "might", "must",
    "thing", "things", "stuff", "lot", "bit", "kind", "sort", "way", "ways",
    "feel", "felt", "feeling", "think", "thought", "thinking", "know", "knew",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")


# ---- helpers ---------------------------------------------------------------

def _entry_text_fields(entry: dict) -> list[tuple[str, str]]:
    """Yield (label, text) pairs from short/long/timeline text-y question values.

    The frontend posts `entry.questions` as a list of question objects with a
    `type` and a `value` field. We pull anything that's reasonably textual.
    """
    pairs: list[tuple[str, str]] = []
    questions = entry.get("questions") or []
    if not isinstance(questions, list):
        return pairs
    for q in questions:
        if not isinstance(q, dict):
            continue
        qtype = q.get("type", "")
        label = q.get("label") or q.get("id") or qtype or "text"
        val = q.get("value")
        if isinstance(val, str) and qtype in {"shorttext", "longtext", "freetext", "text"}:
            text = val.strip()
            if text:
                pairs.append((label, text))
        elif isinstance(val, dict) and qtype == "timeline":
            # timeline: {morning, afternoon, evening}
            for slot in ("morning", "afternoon", "evening"):
                t = val.get(slot)
                if isinstance(t, str) and t.strip():
                    pairs.append((f"{label} {slot}", t.strip()))
    return pairs


def _kind_label(label: str) -> str:
    """Best-effort mapping for the `context` field in english upgrades."""
    label_l = label.lower()
    if "long" in label_l or "free" in label_l or "letter" in label_l or "observation" in label_l:
        return "longtext 'free'"
    if "timeline" in label_l or "morning" in label_l or "afternoon" in label_l or "evening" in label_l:
        return "timeline"
    return "shorttext"


def _find_very_upgrade(text: str) -> tuple[str, str, str] | None:
    """Return (original_substring, suggested_substring, kind) or None."""
    m = re.search(r"\bvery\s+([A-Za-z]+)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    adj = m.group(1).lower()
    swap = _VERY_SWAPS.get(adj)
    if not swap:
        return None
    original = m.group(0)
    return original, swap, "word-choice"


def _find_contraction_upgrade(text: str) -> tuple[str, str, str] | None:
    low = text.lower()
    for long_form, short_form in _CONTRACTIONS:
        idx = low.find(long_form)
        if idx >= 0:
            original = text[idx:idx + len(long_form)]
            # preserve capitalization of first char
            suggested = short_form.capitalize() if original[:1].isupper() else short_form
            return original, suggested, "contraction"
    return None


def _apply_swap(text: str, original: str, suggested: str) -> str:
    """Replace the first occurrence of `original` (case-insensitive) with `suggested`."""
    idx = text.lower().find(original.lower())
    if idx < 0:
        return text.replace(original, suggested, 1)
    return text[:idx] + suggested + text[idx + len(original):]


def _english_upgrades(entry: dict, cap: int = 2) -> list[dict]:
    out: list[dict] = []
    seen_originals: set[str] = set()
    for label, text in _entry_text_fields(entry):
        if len(text) <= 8:
            continue
        for finder, note in (
            (_find_very_upgrade, _VERY_NOTE),
            (_find_contraction_upgrade, _CONTRACTION_NOTE),
        ):
            if len(out) >= cap:
                break
            hit = finder(text)
            if not hit:
                continue
            original_sub, suggested_sub, kind = hit
            key = original_sub.lower()
            if key in seen_originals:
                continue
            seen_originals.add(key)
            out.append({
                "original": original_sub,
                "suggested": _apply_swap(original_sub, original_sub, suggested_sub),
                "note": note,
                "context": _kind_label(label),
                "kind": kind,
            })
        if len(out) >= cap:
            break
    return out


# ---- gamification ----------------------------------------------------------

def _parse_date(s: Any) -> date | None:
    if not isinstance(s, str):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _gamification(entry: dict, history: list[dict]) -> dict:
    today = _parse_date(entry.get("date")) or date.today()
    dates: set[date] = set()
    accents: set[str] = set()

    # Include today's entry in the streak/count set.
    dates.add(today)
    accent_today = (entry.get("accent") or {}).get("name")
    if isinstance(accent_today, str) and accent_today:
        accents.add(accent_today.lower())

    for h in history:
        d = _parse_date(h.get("date"))
        if d:
            dates.add(d)
        acc = (h.get("accent") or {}).get("name") if isinstance(h.get("accent"), dict) else None
        if isinstance(acc, str) and acc:
            accents.add(acc.lower())

    # Consecutive streak ending today.
    streak = 0
    cursor = today
    while cursor in dates:
        streak += 1
        cursor = cursor - timedelta(days=1)

    return {"streak": streak, "color_count": len(accents)}


# ---- patterns --------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _patterns(entry: dict, history: list[dict]) -> list[str]:
    today = _parse_date(entry.get("date")) or date.today()
    cutoff = today - timedelta(days=6)  # last 7 days inclusive
    counter: Counter[str] = Counter()

    def _scan(e: dict) -> None:
        for _, text in _entry_text_fields(e):
            for tok in _tokenize(text):
                if tok in _STOPWORDS or len(tok) < 4:
                    continue
                counter[tok] += 1

    _scan(entry)
    for h in history:
        d = _parse_date(h.get("date"))
        if d is None or d < cutoff or d > today:
            continue
        if d == today:
            continue  # don't double-count if it's today's saved copy
        _scan(h)

    repeated = [(w, n) for w, n in counter.most_common() if n >= 3]
    out: list[str] = []
    for word, n in repeated[:3]:
        out.append(f"you've used '{word}' {n}x this week — try a swap")
    return out


# ---- tomorrow prompt -------------------------------------------------------

def _tomorrow_prompt(entry: dict) -> str:
    d = _parse_date(entry.get("date")) or date.today()
    idx = d.toordinal() % len(_TOMORROW_PROMPTS)
    return _TOMORROW_PROMPTS[idx]


# ---- public ----------------------------------------------------------------

def build_feedback(entry: dict, history: list[dict]) -> dict:
    """Return the feedback JSON shape (see PLAN.md §2.2)."""
    if not isinstance(entry, dict):
        entry = {}
    if not isinstance(history, list):
        history = []

    return {
        "english": _english_upgrades(entry, cap=2),
        "gamification": _gamification(entry, history),
        "patterns": _patterns(entry, history),
        "prompt_for_tomorrow": _tomorrow_prompt(entry),
        "meta": {
            "source": "rules",
            "latency_ms": 0,
        },
    }
