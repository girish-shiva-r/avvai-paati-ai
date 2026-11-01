#!/usr/bin/env bash
# Push the v2 backend to your existing Hugging Face Docker Space.
# Prereqs: pip install -U huggingface_hub ; hf auth login   (token with write access)
# Then in the Space settings add the secret GOOGLE_API_KEY.
set -euo pipefail
SPACE=${1:-GSR-608001/avvaiyar-brain}
cd "$(dirname "$0")/.."
hf upload "$SPACE" . . --repo-type=space \
  --include "Dockerfile" "requirements-api.txt" "api.py" "paatti/*.py" "data/aathichoodi_kb.json" \
  --delete "aathichoodi_english_rich.csv" "requirements.txt" \
  --commit-message "Avvaiyar Paatti v2: curated KB, hybrid retrieval, Gemini selection"
echo "Deployed. Health check: https://${SPACE/\//-}.hf.space/health"
