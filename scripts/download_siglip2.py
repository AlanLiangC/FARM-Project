#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_dest = repo_root / "models" / "siglip2-large-patch16-256"
    parser = argparse.ArgumentParser(description="Download SigLIP2 weights into the repo-local ./models directory.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help=f"Destination directory. Default: {default_dest}",
    )
    parser.add_argument(
        "--repo-id",
        default="google/siglip2-large-patch16-256",
        help="Hugging Face repo id to download from.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit to pin the download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = args.dest.expanduser().resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub", file=sys.stderr)
        print("Install it with: pip install huggingface_hub", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ]

    print(f"Downloading {args.repo_id} to {dest} ...")
    download_kwargs = dict(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    if args.revision:
        download_kwargs["revision"] = args.revision
    snapshot_download(**download_kwargs)
    print(f"Done. Models saved under {dest.parent}")
    print("Launch from inside the repo to use the repo-local default, or export SCENE_GRAPH_MODEL_DIR if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
