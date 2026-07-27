#!/usr/bin/env bash
set -euo pipefail

# Fetch the pretrained weights the pipeline needs into ./models (or --dest).
#
# Everything comes from an official public source - no Hugging Face login and
# no Git LFS required:
#
#   models/yoloe/yoloe-v8l-seg.pt        YOLOE segmentation        (hf.co/jameslahm/yoloe)
#   models/yoloe/yoloe-v8l-seg-pf.pt     YOLOE prompt-free variant (hf.co/jameslahm/yoloe)
#   models/mobileclip/mobileclip_blt.pt  MobileCLIP text encoder   (Apple CDN)
#   models/siglip2-large-patch16-256/    SigLIP2 image/text tower  (hf.co/google, via
#                                        scripts/download_siglip2.py)
#
# The DINOv3 ViT-S/16 merge backbone is already committed at
# models/dinov3-vits16/ (redistributed under the DINOv3 License that ships
# next to it), so a fresh clone runs fully offline. Each model carries its own
# license - see THIRD_PARTY_NOTICES.md before using this project, especially
# commercially.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODELS_DIR="${SCRIPT_DIR}/models"
SKIP_SIGLIP2=0

usage() {
  cat <<'EOF'
Usage: ./bootstrap_models.sh [--dest PATH] [--skip-siglip2]

Downloads YOLOE + MobileCLIP (+ SigLIP2) weights into the models directory.
Pass --skip-siglip2 to defer the 3.4 GB SigLIP2 download (needed for the
SigLIP2 embedding channel; mapping runs without it).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)
      [[ $# -ge 2 ]] || { echo "--dest requires a path" >&2; exit 1; }
      MODELS_DIR="$2"
      shift 2
      ;;
    --skip-siglip2)
      SKIP_SIGLIP2=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

fetch() {
  # fetch <url> <dest-file> <min-bytes>
  local url="$1" dest="$2" min_bytes="$3"
  if [[ -f "$dest" ]] && [[ $(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest") -ge $min_bytes ]]; then
    echo "  [skip] $(basename "$dest") already present"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  [get ] $(basename "$dest") <- $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "${dest}.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${dest}.part" "$url"
  else
    echo "Need curl or wget to download models." >&2
    exit 1
  fi
  mv "${dest}.part" "$dest"
}

echo "Fetching models into ${MODELS_DIR}"

# YOLOE detection/segmentation checkpoints (official YOLOE release on HF).
fetch "https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8l-seg.pt" \
      "${MODELS_DIR}/yoloe/yoloe-v8l-seg.pt" 100000000
fetch "https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8l-seg-pf.pt" \
      "${MODELS_DIR}/yoloe/yoloe-v8l-seg-pf.pt" 100000000

# MobileCLIP-B(LT) text encoder used by YOLOE's open-vocabulary prompts
# (official Apple release).
fetch "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt" \
      "${MODELS_DIR}/mobileclip/mobileclip_blt.pt" 500000000

# SigLIP2 image/text tower (multi-modal embedding channel).
if [[ $SKIP_SIGLIP2 -eq 0 ]]; then
  PY=""
  if command -v python3 >/dev/null 2>&1; then
    PY="python3"
  fi
  if [[ -z "$PY" ]] || ! $PY -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "  [warn] python3 + huggingface_hub not available on the host - skipping SigLIP2."
    echo "         Run later (host or container):"
    echo "         python3 scripts/download_siglip2.py --dest ${MODELS_DIR}/siglip2-large-patch16-256"
  else
    $PY "${SCRIPT_DIR}/scripts/download_siglip2.py" --dest "${MODELS_DIR}/siglip2-large-patch16-256"
  fi
fi

echo ""
echo "Models ready under ${MODELS_DIR}"
echo "(DINOv3 ViT-S/16 is committed in-repo at models/dinov3-vits16.)"
echo ""
echo "Optional, recommended for paper-grade object merging: the gated DINOv3"
echo "ViT-S+/16 backbone. Request access on Hugging Face, then:"
echo "  huggingface-cli download facebook/dinov3-vits16plus-pretrain-lvd1689m --local-dir ${MODELS_DIR}/dinov3-vits16plus"
echo "It is picked up automatically once present (see README)."
