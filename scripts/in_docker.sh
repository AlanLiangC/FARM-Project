#!/usr/bin/env bash
# Run a command inside the running scene-graph docker container.
#
# Runtime work, including pure-Python smoke tests, happens inside the
# container so the host and container Python environments do not drift:
#
#   ./scripts/in_docker.sh python scripts/query_scene_graph.py --pt /data/out/scene.pt --query 'a backpack'
#   ./scripts/in_docker.sh python scripts/view_scene_state.py --pt /data/out/scene.pt
#   ./scripts/in_docker.sh python -m pytest tests/
#
# Notes:
# - First arg ``python`` / ``python3`` is rewritten to the container's
#   ``~/.venv/bin/python``.
# - Working dir is set to the bind-mounted source tree at
#   ``/home/scene_graph/scene_graph`` so relative paths Just Work.
# - Container name defaults to ``scene-graph-batch`` (a long-running
#   keep-alive container created by ``docker compose run --name … -d``);
#   override with ``SG_CONTAINER=foo ./scripts/in_docker.sh …``.

set -euo pipefail

CONTAINER="${SG_CONTAINER:-scene-graph-batch}"
WORKDIR="/home/scene_graph/scene_graph"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' is not running." >&2
    echo "       Start it with:  cd docker && docker compose run -d --name $CONTAINER scene-graph sleep infinity" >&2
    echo "       Or override with:  SG_CONTAINER=<name> ./scripts/in_docker.sh ..." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "usage: $(basename "$0") <command> [args...]" >&2
    echo "       $(basename "$0") python scripts/query_scene_graph.py --pt /data/out/scene.pt --query 'couch'" >&2
    exit 1
fi

# Forward retrieval-relevant env vars that don't have container-side defaults.
# Add more here as new tests need them.
forward_env=()
for var in QWEN3_VL_EMBED_ENABLED LAM_SIGLIP2_LOCAL_ENABLED LAM_SIGLIP2_CKPT VLLM_BASE_URL VLLM_API_KEY VLLM_VL_MODEL VLLM_LLM_MODEL VLLM_MODEL VLLM_TIMEOUT_S VLLM_DISABLE_THINKING VLLM_MAX_TOKENS VLLM_TEMPERATURE VLLM_TOP_P VLLM_CAPTION_MODEL VLLM_CAPTION_DISABLE_THINKING VLLM_CAPTION_PREFIX_WARMUP VLLM_CAPTION_PREFIX_WARMUP_TIMEOUT_S CAPTION_VISUAL_PROMPT_MODE CAPTION_PROMPT_VARIANT VLLM_EMBED_BASE_URL VLLM_EMBED_MODEL VLLM_QWEN3_VL_EMBED_BASE_URL VLLM_QWEN3_VL_EMBED_MODEL CUDA_VISIBLE_DEVICES SCENE_GRAPH_FUSED_COV_RIDGE SCENE_GRAPH_MAX_MERGE_DISTANCE_M SCENE_GRAPH_HELLINGER_MATCH_FLOOR; do
    if [[ -n "${!var:-}" ]]; then
        forward_env+=(-e "$var=${!var}")
    fi
done

# Source ROS + the colcon overlay so rclpy and the colcon-built `mapping`
# package are importable (the offline runner needs both). Rewrite
# ``python`` / ``python3`` to the container's venv python. We use ``bash -lc``
# rather than exec'ing python directly so the env is consistent even when
# callers pass non-python commands.
SETUP_LINES='\
source /opt/ros/humble/setup.bash >/dev/null 2>&1; \
[ -f /home/scene_graph/scene_graph/install/setup.bash ] && \
    source /home/scene_graph/scene_graph/install/setup.bash >/dev/null 2>&1; \
[ -f /tmp/colcon_ws/install/setup.bash ] && \
    source /tmp/colcon_ws/install/setup.bash >/dev/null 2>&1; \
'
case "$1" in
    python|python3)
        shift
        printf -v args_quoted '%q ' "$@"
        inner="${SETUP_LINES} exec /home/scene_graph/.venv/bin/python ${args_quoted}"
        ;;
    *)
        printf -v args_quoted '%q ' "$@"
        inner="${SETUP_LINES} exec ${args_quoted}"
        ;;
esac

exec docker exec -w "$WORKDIR" "${forward_env[@]}" "$CONTAINER" bash -lc "$inner"
