#!/usr/bin/env python3
"""Query a saved scene_state.pt with a free-text description.

Loads a scene graph produced by ``python -m scene_graph.offline.run`` (or
``scripts/run_pipeline.py``), builds a :class:`SceneGraphRetriever` over it,
and prints ranked object clusters for a text query.

Run inside the docker container, with an embedding server reachable
(``./run.sh vllm`` starts one)::

    ./scripts/in_docker.sh python scripts/query_scene_graph.py \\
        --pt /data/out/scene0000_00.pt --query "a red backpack"

Set ``QWEN3_VL_EMBED_ENABLED=0`` to skip the VL-embedding path if only the
text embed server is up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, help="Path to a saved scene_state.pt file.")
    parser.add_argument("--query", required=True, help="Free-text query, e.g. 'a red backpack near the door'.")
    parser.add_argument("--top-k", type=int, default=5, help="How many ranked results to print.")
    parser.add_argument("--spatial-method", default="joint_v1",
                        help="Relational engine (joint_v1 default; unified_soft_w50 = the paper's locked protocol).")
    parser.add_argument("--embedding-only", action="store_true",
                        help="Skip the LLM parse + relational scoring; embedding cluster retrieval only.")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"scene_state.pt not found at {pt_path}")
        return 2

    from scene_graph.llm_utils import EmbedInterface, LLMInterface
    from scene_graph.scene_state_io import load_scene_state

    embedder = EmbedInterface(verbose=False)

    if not args.embedding_only:
        # Relational path (same pipeline as the viser Query panel): LLM parse
        # -> execute_spatial_query. Falls back to embedding clusters when the
        # parser LLM is unreachable.
        try:
            import torch

            from scene_graph.retrieval.spatial_reasoning import execute_spatial_query, parse_query

            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            feature_dim = payload.get("feature_dim") if isinstance(payload, dict) else None
            state = (
                load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
                if feature_dim is not None
                else payload.get("state", payload)
            )
            captions = state.get("object_caption") or []
            llm = LLMInterface(verbose=False)
            t0 = time.time()
            query_graph = parse_query(args.query, llm)
            scored = execute_spatial_query(
                query_graph, state, llm, embedder,
                use_vlm=False, pre_filter_k=-1, max_output_candidates=max(args.top_k, 20),
                raw_query=args.query, retrieval_mode="multi", candidate_pool_mode="active",
                spatial_method=str(args.spatial_method), verbose=False,
            )
            dt = time.time() - t0
            preds = ", ".join(f"{p.name}({', '.join(map(str, p.args))})" for p in query_graph.predicates)
            print(f"\nquery={args.query!r} [{args.spatial_method}] -> parsed [{preds}] in {dt:.2f}s\n")
            for rank, cand in enumerate(scored[: args.top_k], start=1):
                cap = str(captions[cand.object_index] if cand.object_index < len(captions) else "") or "(no caption)"
                cap = (cap[:70] + "…") if len(cap) > 70 else cap
                anchors = ", ".join(str(a) for a in (cand.matched_anchors or {}).values())
                extra = f" anchors=[{anchors}]" if anchors else ""
                print(f"  #{rank} object_id={cand.object_id} score={cand.composite_score:.3f}{extra} {cap!r}")
            return 0
        except Exception as exc:  # noqa: BLE001 - degrade to embedding retrieval
            print(f"Relational path unavailable ({exc}); falling back to embedding retrieval.")

    from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever

    retriever = SceneGraphRetriever.from_scene_state(pt_path, embedder=embedder, verbose=False)
    print(f"Loaded {len(retriever._processor.object_ids)} objects from {pt_path}")

    t0 = time.time()
    result = retriever.retrieve(args.query)
    dt = time.time() - t0

    clusters = result.get("clusters", []) or []
    print(f"\nquery={args.query!r} -> {len(clusters)} clusters in {dt:.2f}s\n")
    for i, cluster in enumerate(clusters[: args.top_k]):
        score = cluster.get("cluster_score", 0.0)
        candidates = cluster.get("candidate_objects", []) or []
        print(f"  #{i + 1} score={score:.3f} ({len(candidates)} candidates)")
        for cand in candidates[:3]:
            caption = str(cand.get("caption") or cand.get("object_caption") or "")
            caption = (caption[:70] + "…") if len(caption) > 70 else caption
            print(f"      object_id={cand.get('object_id')} score={cand.get('final_retrieval_score', 0.0):.3f} {caption!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
