#!/usr/bin/env bash
# journal feedback server starter.
# chmod +x this file before first run:  chmod +x start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Create venv if missing.
if [ ! -d ".venv" ]; then
  echo "creating .venv..."
  python3 -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt

# Load .env from the project root if it exists.
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  echo "loaded env from $ENV_FILE"
else
  echo "WARN: .env not found at $ENV_FILE"
  if [ -t 0 ]; then
    read -r -p "Create .env now for AI feedback? [y/N] " CREATE_ENV
    case "$CREATE_ENV" in
      y|Y|yes|YES)
        read -r -p "Provider [gemini/openai/anthropic] (default: gemini): " PROVIDER_INPUT
        PROVIDER_INPUT="${PROVIDER_INPUT:-gemini}"
        case "$PROVIDER_INPUT" in
          openai) KEY_NAME="OPENAI_API_KEY"; MODEL_ID="gpt-4o" ;;
          anthropic) KEY_NAME="ANTHROPIC_API_KEY"; MODEL_ID="claude-opus-4-7" ;;
          gemini|google|"") PROVIDER_INPUT="gemini"; KEY_NAME="GEMINI_API_KEY"; MODEL_ID="gemini-2.5-flash" ;;
          *)
            echo "WARN: unknown provider '$PROVIDER_INPUT' — using gemini"
            PROVIDER_INPUT="gemini"
            KEY_NAME="GEMINI_API_KEY"
            MODEL_ID="gemini-2.5-flash"
            ;;
        esac
        read -r -s -p "$KEY_NAME: " KEY_VALUE
        echo
        {
          echo "LLM_PROVIDER=$PROVIDER_INPUT"
          echo "MODEL_ID=$MODEL_ID"
          echo "$KEY_NAME=$KEY_VALUE"
          echo "LIFEOS_DEV=0"
        } > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        set -a
        # shellcheck source=/dev/null
        source "$ENV_FILE"
        set +a
        echo "created env at $ENV_FILE"
        ;;
      *)
        echo "AI feedback will use rule-based fallback until you add .env"
        ;;
    esac
  else
    echo "AI feedback will use rule-based fallback until you add .env"
  fi
fi

PROVIDER="${LLM_PROVIDER:-gemini}"

# Install the provider-specific SDK on demand (lazy — only the one in use).
case "$PROVIDER" in
  anthropic)
    pip install -q "anthropic>=0.40"
    KEY_VAR="ANTHROPIC_API_KEY"
    ;;
  openai)
    pip install -q "openai>=1.40"
    KEY_VAR="OPENAI_API_KEY"
    ;;
  gemini|google)
    pip install -q "google-genai>=0.3"
    KEY_VAR="GEMINI_API_KEY"
    if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
      export GEMINI_API_KEY="$GOOGLE_API_KEY"
    fi
    ;;
  *)
    echo "WARN: unknown LLM_PROVIDER=$PROVIDER — will use rule-based fallback only"
    KEY_VAR=""
    ;;
esac

echo "journal feedback server starting on http://localhost:5757"
echo "LLM_PROVIDER: $PROVIDER"
if [ -n "$KEY_VAR" ]; then
  if [ -n "${!KEY_VAR:-}" ]; then
    echo "$KEY_VAR: set"
  else
    echo "$KEY_VAR: missing — will use rule-based fallback"
  fi
fi

exec python feedback_server.py
