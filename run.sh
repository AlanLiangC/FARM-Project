#!/usr/bin/env bash
# Host-side helper for the scene-graph stack.
#
# All runtime workflows live inside the Docker container (see docker/). The uv/
# micromamba host setup is no longer supported. This script manages the vLLM
# captioning/embedding servers (tmux) and launches ROS2 inside the container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"
SESSION="vllm_scene_graph"
HOST="${VLLM_HOST:-127.0.0.1}"

# Pick whichever Compose CLI is available (legacy binary or plugin).
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
elif docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
else
    COMPOSE_CMD=()
fi

# --- Ports ---
PORT_VL8=8000
PORT_EMB=8002
PORT_VL_EMB=8006

# --- Models ---
# Captioning slot: Qwen3.5-9B (multimodal reasoning model, served in non-thinking
# instruct mode via chat_template_kwargs.enable_thinking=False at request time).
MODEL_VL8="Qwen/Qwen3.5-9B"
MODEL_EMB="Qwen/Qwen3-Embedding-0.6B"
MODEL_VL_EMB="Qwen/Qwen3-VL-Embedding-2B"

# --- GPU allocation (override with env vars) ---
GPU_VL8="${GPU_VL8:-0}"
GPU_EMB="${GPU_EMB:-1}"
GPU_VL_EMB="${GPU_VL_EMB:-1}"

# --- vLLM flags ---
# --reasoning-parser qwen3 splits any stray <think>...</think> into the
# `reasoning_content` response field; with enable_thinking=False the model
# shouldn't emit one, but the parser is a safety net.
VL8_FLAGS="--host ${HOST} --port ${PORT_VL8} --served-model-name qwen3.5-9b --max-model-len 3084 --gpu-memory-utilization 0.75 --dtype half --max-num-seqs 5 --max-num-batched-tokens 2048 --enable-chunked-prefill --reasoning-parser qwen3 --disable-log-stats"
EMB_FLAGS="--host ${HOST} --port ${PORT_EMB} --served-model-name qwen3-emb-0.6b --runner pooling --dtype half --max-model-len 512 --gpu-memory-utilization 0.2 --max-num-seqs 5 --max-num-batched-tokens 2048 --enforce-eager --disable-log-stats"
VL_EMB_FLAGS="--host ${HOST} --port ${PORT_VL_EMB} --served-model-name qwen3-vl-emb-2b --runner pooling --dtype half --max-model-len 600 --gpu-memory-utilization 0.7 --max-num-seqs 5 --max-num-batched-tokens 2048 --enforce-eager --disable-log-stats"

# ─────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args...]

Host commands (run from anywhere; paths resolve via SCRIPT_DIR):
  build                       Build the Docker image (scene_graph:latest)
  shell [<dataset_dir>]       Drop into the container, optionally mounting
                              <dataset_dir> as /data inside. You can also set
                              DATASET_DIR=/abs/path instead of passing an arg.

Once inside the container, run the offline pipeline directly:
  python -m scene_graph.offline.run --source sens   --sens-path /data/... --save-path /data/out.pt [--viser] [--covisibility]
  python -m scene_graph.offline.run --source rosbag --bag-path /data/...  --save-path /data/out.pt

Other commands (work on host or inside the container):
  vllm        Start vLLM caption/embedding servers in a tmux session
  ros2        Start vLLM + ros2 launch for online mapping (inside container)
  stop        Stop vLLM tmux session

Environment variables:
  DATASET_DIR   Host dataset dir to bind-mount at /data inside the container
  GPU_VL8       GPU index for the captioning model (Qwen3.5-9B, default: 0)
  GPU_EMB       GPU index for embedding model (default: 1)
  GPU_VL_EMB    GPU index for VL embedding model (default: 1)
  VLLM_HOST     Bind address for vLLM servers (default: 127.0.0.1)
EOF
}

require_compose() {
    if [[ ${#COMPOSE_CMD[@]} -eq 0 ]]; then
        echo "ERROR: neither \`docker-compose\` nor \`docker compose\` is available on this host." >&2
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────
wait_ready() {
    local port="$1"
    local log="$2"
    echo "[wait_ready] waiting for :${port} ..."

    while true; do
        if [[ -f "$log" ]] && grep -q "Application startup complete." "$log"; then
            echo "[wait_ready] :${port} ready"
            return 0
        fi

        if [[ -f "$log" ]] && grep -qE "\] exited code=" "$log"; then
            echo "[wait_ready] ERROR: process on :${port} exited before ready"
            tail -n 40 "$log" || true
            return 1
        fi

        if [[ -f "$log" ]] && grep -qE "EngineCore failed to start|No available memory for the cache blocks|CUDA out of memory|OutOfMemoryError|Traceback" "$log"; then
            echo "[wait_ready] ERROR on :${port}"
            tail -n 40 "$log" || true
            return 1
        fi

        sleep 1
    done
}

# ─────────────────────────────────────────────────────────────
start_vllm() {
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    rm -f /tmp/${SESSION}_*.log

    echo "========================================="
    echo "Starting vLLM models"
    echo "========================================="

    echo "Starting Qwen3.5-9B on port ${PORT_VL8} (GPU ${GPU_VL8})..."
    tmux new-session -d -s "$SESSION" -n vl8 \
        "bash -lc 'set -x; export CUDA_VISIBLE_DEVICES=${GPU_VL8}; vllm serve ${MODEL_VL8} ${VL8_FLAGS} 2>&1 | tee /tmp/${SESSION}_vl8.log; ec=\${PIPESTATUS[0]}; echo \"[vl8] exited code=\$ec\"; read -rp \"Press Enter...\"'"

    wait_ready "${PORT_VL8}" "/tmp/${SESSION}_vl8.log"

    echo "Starting Qwen3-Embedding-0.6B on port ${PORT_EMB} (GPU ${GPU_EMB})..."
    tmux new-window -t "$SESSION" -n emb \
        "bash -lc 'set -x; export CUDA_VISIBLE_DEVICES=${GPU_EMB}; vllm serve ${MODEL_EMB} ${EMB_FLAGS} 2>&1 | tee /tmp/${SESSION}_emb.log; ec=\${PIPESTATUS[0]}; echo \"[emb] exited code=\$ec\"; read -rp \"Press Enter...\"'"

    wait_ready "${PORT_EMB}" "/tmp/${SESSION}_emb.log"

    echo "Starting Qwen3-VL-Embedding-2B on port ${PORT_VL_EMB} (GPU ${GPU_VL_EMB})..."
    tmux new-window -t "$SESSION" -n vl_emb \
        "bash -lc 'set -x; export CUDA_VISIBLE_DEVICES=${GPU_VL_EMB}; vllm serve ${MODEL_VL_EMB} ${VL_EMB_FLAGS} 2>&1 | tee /tmp/${SESSION}_vl_emb.log; ec=\${PIPESTATUS[0]}; echo \"[vl_emb] exited code=\$ec\"; read -rp \"Press Enter...\"'"

    wait_ready "${PORT_VL_EMB}" "/tmp/${SESSION}_vl_emb.log"

    echo "========================================="
    echo "All vLLM models ready"
    echo "========================================="
    echo "  Port ${PORT_VL8} — Qwen3.5-9B (captioning, non-thinking instruct mode)"
    echo "  Port ${PORT_EMB} — Qwen3-Embedding-0.6B (text embeddings)"
    echo "  Port ${PORT_VL_EMB} — Qwen3-VL-Embedding-2B (VL embeddings)"
    echo ""
    echo "Logs: /tmp/${SESSION}_*.log"
    echo "Attach: tmux attach -t ${SESSION}"
}

# ─────────────────────────────────────────────────────────────
export_vllm_env() {
    export VLLM_BASE_URL="http://localhost:${PORT_VL8}/v1"
    export VLLM_EMBED_BASE_URL="http://localhost:${PORT_EMB}/v1"
    export VLLM_QWEN3_VL_EMBED_BASE_URL="http://localhost:${PORT_VL_EMB}/v1"
    export SCENE_GRAPH_MODEL_DIR="${SCRIPT_DIR}/models"
}

# ─────────────────────────────────────────────────────────────
run_ros2() {
    start_vllm
    export_vllm_env
    echo ""
    echo "Launching ROS2 scene graph..."
    ros2 launch mapping scenegraph_validation_exploration.launch.py "$@"
}

# ─────────────────────────────────────────────────────────────
stop_vllm() {
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped ${SESSION}." || echo "No active session."
}

# ─────────────────────────────────────────────────────────────
case "${1:-}" in
    build)
        require_compose
        exec "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" build
        ;;
    shell)
        require_compose
        shift || true
        dataset_arg="${1:-${DATASET_DIR:-}}"
        volume_flags=()
        if [[ -n "$dataset_arg" ]]; then
            if [[ ! -d "$dataset_arg" ]]; then
                echo "ERROR: dataset directory '$dataset_arg' does not exist." >&2
                exit 1
            fi
            dataset_abs="$(cd "$dataset_arg" && pwd)"
            volume_flags=(-v "${dataset_abs}:/data")
            echo "[run.sh] mounting ${dataset_abs} -> /data"
        fi
        exec "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" run --rm "${volume_flags[@]}" scene-graph
        ;;
    vllm)
        start_vllm
        ;;
    ros2)
        shift
        run_ros2 "$@"
        ;;
    stop)
        stop_vllm
        ;;
    *)
        usage
        exit 1
        ;;
esac
