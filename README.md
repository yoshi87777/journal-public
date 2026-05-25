# journal

A daily English journaling app with inline AI feedback. Runs entirely on
your machine — your entries live in plain JSON files on disk, and only the
feedback step calls out to an LLM.

This is a slimmed-down, local-only build. No cloud database, no public
tunnel, no deployment scripts.

## What you get

| Page | What it's for |
|---|---|
| `journal/today.html` | The daily form. Date-seeded prompts (same prompts on refresh, new ones tomorrow). Inline feedback after save. |
| `journal/write.html` | Long-form writing canvas with autosave, session timer, and live english/reflect rails. |
| `journal/archive.html` | Past entries as cards. Mood sparkline, color mosaic, tag cloud, "on this day" callout. |
| `journal/english.html` | Phrasing-upgrade history + recurring-pattern frequency. |

The "feedback" half is provided by a tiny local FastAPI server on
`http://localhost:5757`. It writes your entry to disk, asks an LLM for
feedback, writes the feedback to disk, and returns it to the page.

## Quick start

### 1. Get an API key

Default provider is Gemini (free tier).
Sign up at <https://ai.google.dev/gemini-api> and grab a key.

(If you'd rather use Anthropic or OpenAI, see `.env.example`.)

### 2. Configure

```bash
cp .env.example .env
# then edit .env and paste your GEMINI_API_KEY
```

### 3. Run the server

```bash
cd journal/scripts
chmod +x start.sh
./start.sh
```

The script creates a Python venv on first run, installs deps, and starts
the server. You should see:

```
journal feedback server starting on http://localhost:5757
LLM_PROVIDER: gemini
GEMINI_API_KEY: set
```

### 4. Open a page

Open `journal/today.html` directly in your browser (double-click the file,
or `open journal/today.html` on macOS). The page auto-detects the running
server and POSTs your entry on save.

That's it. Write the form, hit `⌘+S`, watch feedback render inline.

## Without an API key

If `GEMINI_API_KEY` is missing, the server falls back to
`feedback_rules.py` — a rule-based feedback engine. Gamification still
works (streaks, color count, badges) but you lose the `reflect`
observations and english upgrades.

## Where your data lives

- `journal/entries/<YYYY-MM-DD>.json` — what you wrote
- `journal/feedback/<YYYY-MM-DD>.json` — the LLM's response
- `journal/conversations/<YYYY-MM-DD>.json` — any reply threads you start
  from a feedback card

These directories are git-ignored. They're created on first save.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `⌘+S` | Save today |
| `esc` | Reroll prompts (new date-seed) |
| `⌘+/` | Open shortcut sheet |

## Dark / light

A toggle lives in the corner. Defaults to system preference.

## Provider switching

In `.env`, set `LLM_PROVIDER` to one of `gemini` / `anthropic` / `openai`
and provide the matching key. `start.sh` installs the right SDK on
demand. The provider-agnostic seam lives in `journal/scripts/insight.py`.

## File layout

```
.
├── .env.example           # copy → .env
├── .gitignore
├── README.md
└── journal/
    ├── today.html         # daily form
    ├── write.html         # long-form canvas
    ├── archive.html       # past entries
    ├── english.html       # phrasing history
    ├── _tokens.css        # design tokens (don't edit)
    ├── _accent.js         # daily accent picker (don't edit)
    ├── _api.js            # localhost shim
    └── scripts/
        ├── start.sh             # boots the venv + server
        ├── feedback_server.py   # FastAPI on :5757
        ├── insight.py           # LLM seam (provider-agnostic)
        ├── feedback_rules.py    # rule-based fallback
        ├── requirements.txt
        └── vendor/
            └── llm_client.py    # vendored chat() supporting all 3 providers
```

## Notes

- Server binds `127.0.0.1` only. Never exposes outside your machine.
- Request bodies are capped at 100KB. Host header is validated.
- API keys are read from env. Never logged, never echoed.
- Entry content is never logged. Only method/path/status/latency.
