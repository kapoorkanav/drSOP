#!/usr/bin/env bash
# Clones the official RETFound_MAE repo (needed for models_vit.py + position-embedding
# interpolation code, which our RetfoundEncoder imports at runtime). Run once on the VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/RETFound_MAE"

if [ -d "$DEST" ]; then
  echo "third_party/RETFound_MAE already exists, skipping clone."
else
  git clone https://github.com/rmaphoh/RETFound_MAE.git "$DEST"
fi

echo ""
echo "Next steps (manual, gated downloads):"
echo "1. Download RETFound_cfp_weights.pth from the official RETFound_MAE README"
echo "   (form-gated; the repo README links to the request form) and place it at:"
echo "   $ROOT/weights/RETFound_cfp_weights.pth"
echo "2. Get credentialed access to BRSET on PhysioNet and download labels.csv +"
echo "   fundus_photos/ into:"
echo "   $ROOT/data/labels.csv"
echo "   $ROOT/data/fundus_photos/"
