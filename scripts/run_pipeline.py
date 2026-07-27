#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LOGGER = logging.getLogger("mapping.pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline mapping pipeline (keyframe gate + YOLOE segmentation).")
    parser.add_argument("--config", type=Path, required=True, help="YAML config (e.g. configs/replica.yaml)")
    parser.add_argument("--dataset-root", type=Path, help="Override dataset.base_dir")
    parser.add_argument("--sequence", type=str, help="Override dataset.sequence (e.g. office0)")
    parser.add_argument("--dataset-device", type=str, default="cuda:0", help="Device for dataset tensors")
    parser.add_argument("--model-id", type=str, default="yoloe-v8l", help="YOLOE model id")
    parser.add_argument("--vocab-file", type=Path, default="configs/yoloe_vocabulary.txt", help="Vocabulary file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (square)")
    parser.add_argument("--conf", type=float, default=0.25, help="Segmentation confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="Segmentation IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Segmentation device (defaults to auto)")
    parser.add_argument(
        "--mask-erosion-px",
        type=int,
        default=3,
        help="Erode segmentation masks by N pixels before 3D stats (0 disables).",
    )
    parser.add_argument(
        "--mahalanobis-thresh",
        type=float,
        default=2.0,
        help="Mahalanobis distance threshold for outlier rejection (<=0 disables).",
    )
    parser.add_argument("--log-every", type=int, default=25, help="Log every N frames even if rejected.")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of frames processed per iteration.")
    parser.add_argument("--log-time", action="store_true", help="Log processing time per frame.")
    parser.add_argument("--viser", action="store_true", help="Serve the live map in a browser.")
    parser.add_argument("--debug", action="store_true", help="Enable additional debug bookkeeping.")
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Enable caption worker; otherwise placeholder captions are used.",
    )
    parser.add_argument(
        "--vis-segmentation",
        type=Path,
        help="Directory where segmentation visualizations will be written.",
    )
    parser.add_argument(
        "--dino",
        action="store_true",
        help="Use DINOv3 ViT-S+/16 features for each segment instead of YOLOE embeddings.",
    )
    parser.add_argument(
        "--dino-weights-path",
        type=Path,
        default=None,
        help="Local directory containing the DINOv3 ViT-S+/16 weights.",
    )
    return parser.parse_args()


def configure_dataset(config, args: argparse.Namespace):
    from scene_graph.datasets import get_dataset
    from scene_graph.config import PipelineConfig

    # Accept either a typed PipelineConfig or a legacy raw dict
    if isinstance(config, PipelineConfig):
        dataset_cfg = dataclasses.asdict(config.dataset)
    else:
        dataset_cfg = dict(config.get("dataset", {}))

    if args.dataset_root is not None:
        dataset_cfg["base_dir"] = str(args.dataset_root)
    if args.sequence is not None:
        dataset_cfg["sequence"] = args.sequence
    if not dataset_cfg.get("base_dir") or not dataset_cfg.get("sequence"):
        raise ValueError("dataset.base_dir and dataset.sequence must be provided (config or CLI overrides).")
    dataset_cfg.setdefault("device", args.dataset_device)
    return get_dataset(dataset_cfg)


def main() -> None:
    from scene_graph.captioning.services import CaptionManager
    from scene_graph.map_update.models import initialize_scene_graph_state
    from scene_graph.mapping_util import iter_batches
    from scene_graph.pipeline import FrameBatch, PipelineOrchestrator
    from scene_graph.segmentation import DINOFeaturesExtractor, YOLOESegmenter
    from scene_graph.storage.image_save_worker import ImageSaveWorker
    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer

    from scene_graph.config import PipelineConfig

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    config = PipelineConfig.from_yaml(args.config)
    dataset = configure_dataset(config, args)
    # CLI flags override the typed config where provided
    seg_cfg = config.segmentation
    if args.model_id != "yoloe-v8l":
        seg_cfg.model_id = args.model_id
    if str(args.vocab_file) != "configs/yoloe_vocabulary.txt":
        seg_cfg.vocab_file = str(args.vocab_file)
    if args.imgsz != 640:
        seg_cfg.imgsz = args.imgsz
    if args.conf != 0.25:
        seg_cfg.conf_thres = args.conf
    if args.iou != 0.5:
        seg_cfg.iou_thres = args.iou
    if args.mask_erosion_px != 3:
        seg_cfg.mask_erosion_px = args.mask_erosion_px
    if args.mahalanobis_thresh != 2.0:
        seg_cfg.mahalanobis_thresh = args.mahalanobis_thresh
    if args.dino:
        seg_cfg.use_dino_features = True
    if args.device:
        seg_cfg.device = args.device

    dino_extractor = None
    if seg_cfg.use_dino_features:
        dino_cfg = seg_cfg.dino
        # CLI override for weights path
        if args.dino_weights_path is not None:
            dino_cfg.weights_path = str(args.dino_weights_path)
        dino_extractor = DINOFeaturesExtractor(
            model=dino_cfg.model,
            load_size=dino_cfg.load_size,
            weights_path=dino_cfg.weights_path,
            device=seg_cfg.device,
        )
    segmenter = YOLOESegmenter(
        model_id=seg_cfg.model_id,
        vocab_file=seg_cfg.vocab_file,
        imgsz=seg_cfg.imgsz,
        conf_thres=seg_cfg.conf_thres,
        iou_thres=seg_cfg.iou_thres,
        device=seg_cfg.device,
        use_dino_features=seg_cfg.use_dino_features,
        dino_extractor=dino_extractor,
        mask_erosion_px=seg_cfg.mask_erosion_px,
        mahalanobis_thresh=seg_cfg.mahalanobis_thresh,
        vis_segmentation_dir=args.vis_segmentation,
    )

    # initialize database variables
    scene_state = initialize_scene_graph_state(segmenter.feature_dim, segmenter.device)
    viser_visualizer = PipelineViserVisualizer(enabled=args.viser)
    caption_manager = CaptionManager(
        scene_state=scene_state,
        enabled=args.caption,
        debug=args.debug,
        caption_device="cuda:0",
    )
    if caption_manager.enabled:
        caption_manager.maybe_start_worker()
        # Eagerly download/initialize Qwen3-VL before processing frames
        caption_manager.warm_up_model()

    image_storage_dir = Path(config.storage.image_dir)
    image_storage_dir.mkdir(parents=True, exist_ok=True)
    image_save_worker = ImageSaveWorker()

    dataset_name = getattr(dataset, "name", "dataset")
    dataset_slug = dataset_name.replace(os.sep, "_")
    LOGGER.info("Dataset frames: %d • Sequence: %s", len(dataset), dataset_name)

    batch_size = max(1, args.batch_size)

    orchestrator = PipelineOrchestrator(
        segmenter,
        scene_state,
        caption_manager,
        image_save_worker,
        image_storage_dir=image_storage_dir,
        dataset_slug=dataset_slug,
        viser_visualizer=viser_visualizer if args.viser else None,
        debug=args.debug,
        filtering_config=config.filtering,
    )

    try:
        for indices, colors, depths, intrinsics_batch, poses in iter_batches(dataset, batch_size):
            batch = FrameBatch(
                colors=list(colors),
                depths=list(depths),
                intrinsics=list(intrinsics_batch),
                poses_world=list(poses),
            )
            orchestrator.process_batch(batch)
    finally:
        orchestrator.flush_captions()
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
