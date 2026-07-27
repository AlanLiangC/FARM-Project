#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-${ROOT}/FARM-Project-source.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

git -C "$ROOT" submodule update --init --recursive
mkdir -p "$STAGE/FARM-Project/third_party/yoloe"
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE/FARM-Project"
git -C "$ROOT/third_party/yoloe" archive HEAD | tar -x -C "$STAGE/FARM-Project/third_party/yoloe"
tar -czf "$OUTPUT" -C "$STAGE" FARM-Project
echo "$OUTPUT"
