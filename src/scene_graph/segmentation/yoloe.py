from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE
from ultralytics.utils import ops

from scene_graph.runtime_paths import find_model_file, find_package_file

from .interfaces import SegmentationBackend
from .models import SegmentationOutput
from .dino import DEFAULT_MODEL as DEFAULT_DINO_MODEL
from .dino import DINOFeaturesExtractor
from .visualization import SegmentationVisualizer

LOGGER = logging.getLogger(__name__)


def _resolve_weights(model_id: str, prompt_free: bool) -> Path:
    """Return a local path to the requested checkpoint, downloading from HF if necessary."""
    suffix = "-seg-pf.pt" if prompt_free else "-seg.pt"
    local_path = find_model_file(f"{model_id}{suffix}", "yoloe")

    if local_path is not None:
        return local_path
    LOGGER.info(
        "Downloading %s weights for YOLOE (%s)",
        "prompt-free" if prompt_free else "base",
        model_id,
    )
    filename = f"{model_id}{suffix}"
    downloaded = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    return Path(downloaded)


def _load_vocab_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


class YOLOESegmenter(SegmentationBackend):
    """
    Prompt-free YOLOE segmentation backend with on-the-fly 3D statistics.

    The implementation mirrors the logic in ``yoloe_test.py`` but exposes a programmatic API.
    """

    def __init__(
        self,
        model_id: str = "yoloe-v8l",
        vocab_file: Path | None = None,
        imgsz: int | tuple[int, int] = 640,
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
        device: str | None = None,
        use_dino_features: bool = False,
        dino_extractor: DINOFeaturesExtractor | None = None,
        max_det: int = 200,
        min_mask_pixels: int = 50,
        min_depth_points: int = 50,
        mask_erosion_px: int = 3,
        mahalanobis_thresh: float = 2.0,
        depth_mode_filter_enabled: bool = True,
        depth_mode_k_mad: float = 3.0,
        depth_mode_min_mad_m: float = 0.03,
        vis_segmentation_dir: Path | str | None = None,
        timing_enabled: bool = False,
        # debug_save_detections: bool = False,
        # debug_save_protos: bool = False,
        # log_time: bool = False
    ) -> None:
        self.model_id = model_id
        default_vocab_file = find_package_file("configs", "yoloe_vocabulary.txt")
        self.vocab_file = Path(vocab_file) if vocab_file else Path(default_vocab_file or "configs/yoloe_vocabulary.txt")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._use_dino_features = bool(use_dino_features)
        self.dino_extractor = dino_extractor
        self._extra_vocab_cache: "OrderedDict[tuple[str, ...], torch.Tensor]" = OrderedDict()
        self._extra_vocab_cache_max = 5
        self._extra_vocab_model: YOLOE | None = None
        self._base_ckpt: Path | None = None
        if isinstance(imgsz, int):
            self.imgsz = (imgsz, imgsz)
        else:
            self.imgsz = tuple(imgsz)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.min_mask_pixels = min_mask_pixels
        self.min_depth_points = min_depth_points
        self.mask_erosion_px = max(0, int(mask_erosion_px))
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.depth_mode_filter_enabled = bool(depth_mode_filter_enabled)
        self.depth_mode_k_mad = float(depth_mode_k_mad)
        self.depth_mode_min_mad_m = float(depth_mode_min_mad_m)
        self._mahalanobis_eps = 1e-6
        self._pixel_grid_cache: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
        self._timing_enabled = bool(timing_enabled)
        self.names: list[str] = []
        self.model = self._init_prompt_free_model()
        head = self.model.model.model[-1]
        self._head = head

        # Feature dimension of the classification embedding (cls_feat)
        # YOLOESegment / YOLOEDetect expose this as `embed`
        self.feature_dim = getattr(head, "embed", 512)

        self._visualizer = SegmentationVisualizer(vis_segmentation_dir)

        # Hooks to capture *classification embeddings* = cv3 outputs (cls_feat)
        # one tensor per FPN level (P3, P4, P5)
        self._embedding_features: list[torch.Tensor | None] = [None] * len(head.cv3)

        if hasattr(head, "cv3"):
            for i, cls_conv in enumerate(head.cv3):

                def _cv3_hook_factory(idx: int):
                    def _cv3_hook(module, inputs, outputs):
                        # outputs shape: [B, embed_dim, H, W]
                        self._embedding_features[idx] = outputs.detach()

                    return _cv3_hook

                cls_conv.register_forward_hook(_cv3_hook_factory(i))

            LOGGER.info(
                "Registered hooks on YOLOE head.cv3 (cls_feat) for embeddings (dim=%d).",
                self.feature_dim,
            )
        else:
            LOGGER.warning("YOLOE head has no 'cv3'; embeddings will fall back to zeros.")

        # Hooks to capture LRPC cls/region feature maps (pre-vocab) and masks
        self._lrpc_num_levels = len(head.lrpc) if hasattr(head, "lrpc") else 0
        self._lrpc_cls_features: list[torch.Tensor | None] = [None] * self._lrpc_num_levels
        self._lrpc_masks: list[torch.Tensor | None] = [None] * self._lrpc_num_levels
        if self._lrpc_num_levels:
            if hasattr(head.lrpc[0], "vocab") and hasattr(head.lrpc[0].vocab, "in_features"):
                self.feature_dim = int(head.lrpc[0].vocab.in_features)

            for i, lrpc_head in enumerate(head.lrpc):

                def _lrpc_hook_factory(idx: int):
                    def _lrpc_hook(module, inputs, outputs):
                        cls_feat = inputs[0]
                        mask = outputs[1] if isinstance(outputs, tuple) else None
                        self._lrpc_cls_features[idx] = cls_feat.detach()
                        self._lrpc_masks[idx] = mask.detach() if mask is not None else None

                    return _lrpc_hook

                lrpc_head.register_forward_hook(_lrpc_hook_factory(i))

            LOGGER.info(
                "Registered hooks on YOLOE LRPC heads for cls features (dim=%d).",
                self.feature_dim,
            )

        if self._use_dino_features:
            if self.dino_extractor is None:
                self.dino_extractor = DINOFeaturesExtractor(
                    model=DEFAULT_DINO_MODEL,
                    load_size=512,
                    device=self.device,
                )
            if self.dino_extractor.hidden_size is None:
                cfg = getattr(self.dino_extractor, "model", None)
                cfg = getattr(cfg, "config", None)
                maybe_hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
                if maybe_hidden is None and cfg is not None:
                    hs_list = getattr(cfg, "hidden_sizes", None)
                    if hs_list:
                        try:
                            maybe_hidden = int(hs_list[-1])
                        except Exception:
                            maybe_hidden = None
                if maybe_hidden is not None:
                    self.dino_extractor.hidden_size = int(maybe_hidden)
            if self.dino_extractor.hidden_size is None:
                raise RuntimeError("Unable to determine DINO hidden size; ensure weights are available.")
            self.feature_dim = int(self.dino_extractor.hidden_size)

    def _init_prompt_free_model(self) -> YOLOE:
        print(f"vocab file path: {self.vocab_file}")
        vocab_names = _load_vocab_list(self.vocab_file)
        self._base_ckpt = _resolve_weights(self.model_id, prompt_free=False)
        base = YOLOE(str(self._base_ckpt))
        vocab = base.get_vocab(vocab_names)
        del base
        pf_ckpt = _resolve_weights(self.model_id, prompt_free=True)
        model = YOLOE(str(pf_ckpt))
        model.set_vocab(vocab, names=vocab_names)
        head = model.model.model[-1]
        head.is_fused = True
        # Keep the internal head threshold very low and rely on
        # postprocess (NMS `conf_thres`) for filtering, matching yoloe_test.py.
        head.conf = 0.001
        head.max_det = self.max_det
        model.to(self.device)
        model.eval()
        self.names = vocab_names
        return model

    def _cache_get(self, key: tuple[str, ...]) -> torch.Tensor | None:
        value = self._extra_vocab_cache.get(key)
        if value is None:
            return None
        self._extra_vocab_cache.move_to_end(key)
        return value

    def _cache_put(self, key: tuple[str, ...], value: torch.Tensor) -> None:
        self._extra_vocab_cache[key] = value
        self._extra_vocab_cache.move_to_end(key)
        while len(self._extra_vocab_cache) > self._extra_vocab_cache_max:
            _, old = self._extra_vocab_cache.popitem(last=False)
            del old

    def _get_extra_text_emb(self, names: Sequence[str]) -> torch.Tensor:
        if not names:
            return torch.empty((0, self.feature_dim), device=self.device, dtype=torch.float32)
        key = tuple(names)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        use_self_model = hasattr(self.model, "get_vocab")
        if use_self_model:
            try:
                head = self.model.model.model[-1]
            except Exception:
                head = None
                use_self_model = False
            if head is not None and getattr(head, "is_fused", False):
                use_self_model = False

        if use_self_model:
            vocab_source = self.model
        else:
            if self._extra_vocab_model is None:
                base_ckpt = self._base_ckpt or _resolve_weights(self.model_id, prompt_free=False)
                self._extra_vocab_model = YOLOE(str(base_ckpt))
                self._extra_vocab_model.to(self.device)
                self._extra_vocab_model.eval()
            vocab_source = self._extra_vocab_model

        with torch.no_grad():
            text_emb = None
            if hasattr(vocab_source, "model") and hasattr(vocab_source.model, "get_text_pe"):
                try:
                    text_emb = vocab_source.model.get_text_pe(list(names), cache_clip_model=True)
                except TypeError:
                    text_emb = vocab_source.model.get_text_pe(list(names))
            elif hasattr(vocab_source, "get_text_pe"):
                text_emb = vocab_source.get_text_pe(list(names))

            if not isinstance(text_emb, torch.Tensor):
                raise RuntimeError("YOLOE text embedding unavailable; ensure text model weights are installed.")

            emb = text_emb
            if emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)
            elif emb.ndim != 2:
                raise RuntimeError(f"Unexpected text embedding shape {tuple(emb.shape)} from YOLOE.")

            emb = emb.to(device=self.device, dtype=torch.float32)
            if emb.shape[1] != self.feature_dim:
                proj_model = self._extra_vocab_model or vocab_source
                proj_head = None
                try:
                    proj_head = proj_model.model.model[-1]
                except Exception:
                    proj_head = None
                if proj_head is not None and hasattr(proj_head, "cv3"):
                    projected = None
                    for idx, cls_head in enumerate(proj_head.cv3):
                        if not isinstance(cls_head, torch.nn.Sequential):
                            continue
                        conv = cls_head[-1]
                        if not isinstance(conv, torch.nn.Conv2d):
                            continue
                        weight = conv.weight.squeeze(-1).squeeze(-1)
                        if weight.ndim != 2:
                            continue
                        if weight.shape[0] != emb.shape[1] or weight.shape[1] != self.feature_dim:
                            continue
                        scale = None
                        if hasattr(proj_head, "cv4") and idx < len(proj_head.cv4):
                            logit_scale = getattr(proj_head.cv4[idx], "logit_scale", None)
                            if isinstance(logit_scale, torch.Tensor):
                                scale = logit_scale.exp()
                        if scale is not None:
                            emb = emb * scale
                        projected = emb @ weight
                        break
                    if projected is not None:
                        emb = projected
                    else:
                        LOGGER.warning(
                            "Extra vocab embedding dim (%d) != visual feature dim (%d); unable to project.",
                            int(emb.shape[1]),
                            int(self.feature_dim),
                        )
            emb = F.normalize(emb, dim=1).detach()

        self._cache_put(key, emb)
        return emb

    @staticmethod
    def _prepare_depth(depth: torch.Tensor) -> torch.Tensor:
        depth = depth.squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected depth to be (H,W) or (H,W,1), got shape {tuple(depth.shape)}")
        return depth

    def _preprocess(
        self, color: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float], tuple[int, int]]:
        """Letterbox the input tensor directly on the target device."""
        if color.ndim == 3:
            if color.shape[0] == 3 and color.shape[-1] != 3:
                color_chw = color
            elif color.shape[-1] == 3:
                color_chw = color.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected color tensor shape {tuple(color.shape)}")
        elif color.ndim == 4 and color.shape[0] == 1:
            color_chw = color.squeeze(0)
            if color_chw.shape[0] != 3:
                raise ValueError(f"Unexpected batched color tensor shape {tuple(color.shape)}")
        else:
            raise ValueError(f"Color tensor must be HWC or CHW, got {tuple(color.shape)}")

        orig_h, orig_w = int(color_chw.shape[1]), int(color_chw.shape[2])
        tensor = color_chw.unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        target_h, target_w = self.imgsz
        gain = min(target_h / orig_h, target_w / orig_w)
        resized_h = int(round(orig_h * gain))
        resized_w = int(round(orig_w * gain))
        if (resized_h, resized_w) != (orig_h, orig_w):
            tensor = F.interpolate(
                tensor,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )

        dh = target_h - resized_h
        dw = target_w - resized_w
        top = int(round(dh / 2 - 0.1))
        bottom = int(round(dh / 2 + 0.1))
        left = int(round(dw / 2 - 0.1))
        right = int(round(dw / 2 + 0.1))
        tensor = F.pad(tensor, (left, right, top, bottom), value=114.0 / 255.0)

        ratio = (gain, gain)
        pad = (dw / 2, dh / 2)
        return tensor, ratio, pad, (orig_h, orig_w)

    @staticmethod
    def _letterbox_params(
        orig_h: int, orig_w: int, target_h: int, target_w: int
    ) -> tuple[float, int, int, int, int, int, int]:
        gain = min(target_h / float(orig_h), target_w / float(orig_w))
        resized_h = int(round(orig_h * gain))
        resized_w = int(round(orig_w * gain))
        dh = target_h - resized_h
        dw = target_w - resized_w
        top = int(round(dh / 2 - 0.1))
        bottom = int(round(dh / 2 + 0.1))
        left = int(round(dw / 2 - 0.1))
        right = int(round(dw / 2 + 0.1))
        return gain, resized_h, resized_w, left, right, top, bottom

    def _letterbox_depth(
        self,
        depth_map: torch.Tensor,
        orig_shape: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, float, tuple[int, int, int, int]]:
        """Letterbox a depth map to target_hw on self.device (pads with 0 depth)."""
        if depth_map.ndim != 2:
            raise ValueError(f"Expected depth_map to be (H,W), got shape {tuple(depth_map.shape)}")
        orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        gain, resized_h, resized_w, left, right, top, bottom = self._letterbox_params(
            orig_h, orig_w, target_h, target_w
        )
        out = depth_map
        if (resized_h, resized_w) != (orig_h, orig_w):
            out = (
                F.interpolate(
                    out.unsqueeze(0).unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )
        if left or right or top or bottom:
            out = F.pad(out.unsqueeze(0).unsqueeze(0), (left, right, top, bottom), value=0.0).squeeze(0).squeeze(0)
        return out, float(gain), (left, top, right, bottom)

    def _get_pixel_grid(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (height, width, device)
        if key not in self._pixel_grid_cache:
            y_coords = torch.arange(height, device=device, dtype=dtype)
            x_coords = torch.arange(width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._pixel_grid_cache[key] = (grid_y, grid_x)
        return self._pixel_grid_cache[key]

    def _erode_masks(self, masks: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0 or masks.numel() == 0:
            return masks
        kernel = int(radius) * 2 + 1
        masks_f = masks.to(dtype=torch.float32)
        inv = 1.0 - masks_f
        dilated = F.max_pool2d(inv.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius)
        eroded = (1.0 - dilated).squeeze(1)
        return (eroded > 0.5).to(torch.bool)

    def _compute_weighted_stats(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = weights.sum(dim=(1, 2))
        sx = (X * weights).sum(dim=(1, 2))
        sy = (Y * weights).sum(dim=(1, 2))
        sz = (Z * weights).sum(dim=(1, 2))

        n_safe = n.clamp_min(1.0)
        inv_n = 1.0 / n_safe
        mx, my, mz = sx * inv_n, sy * inv_n, sz * inv_n
        means = torch.stack([mx, my, mz], dim=1)

        XX = X * X
        YY = Y * Y
        ZZ = Z * Z
        XY = X * Y
        XZ = X * Z
        YZ = Y * Z

        sxx = (XX * weights).sum(dim=(1, 2))
        syy = (YY * weights).sum(dim=(1, 2))
        szz = (ZZ * weights).sum(dim=(1, 2))
        sxy = (XY * weights).sum(dim=(1, 2))
        sxz = (XZ * weights).sum(dim=(1, 2))
        syz = (YZ * weights).sum(dim=(1, 2))

        nm1 = (n - 1.0).clamp_min(1.0)
        c_xx = (sxx - sx * sx * inv_n) / nm1
        c_yy = (syy - sy * sy * inv_n) / nm1
        c_zz = (szz - sz * sz * inv_n) / nm1
        c_xy = (sxy - sx * sy * inv_n) / nm1
        c_xz = (sxz - sx * sz * inv_n) / nm1
        c_yz = (syz - sy * sz * inv_n) / nm1
        cov6 = torch.stack([c_xx, c_xy, c_xz, c_yy, c_yz, c_zz], dim=1)
        return n, means, cov6

    def _compute_mask_medians(
        self,
        X: torch.Tensor,  # (M,H,W)
        Y: torch.Tensor,  # (M,H,W)
        Z: torch.Tensor,  # (M,H,W)
        weights: torch.Tensor,  # (M,H,W) float or bool; inliers are >0
        min_points: int,
    ) -> torch.Tensor:
        """
        Coordinate-wise median of XYZ over inlier pixels for each detection.
        Returns (M,3) with NaNs for detections with too few points.
        """
        device = X.device
        dtype = X.dtype
        M = int(weights.shape[0])

        med = torch.full((M, 3), float("nan"), device=device, dtype=dtype)

        # Note: per-detection loop is fine for M<=200; avoids building huge ragged tensors.
        for i in range(M):
            m = weights[i] > 0  # bool mask of inlier pixels
            if int(m.sum().item()) < int(min_points):
                continue

            # masked_select returns 1D tensor of selected pixels
            xi = X[i].masked_select(m)
            yi = Y[i].masked_select(m)
            zi = Z[i].masked_select(m)

            # torch.median returns a scalar for 1D tensors
            med[i, 0] = xi.median()
            med[i, 1] = yi.median()
            med[i, 2] = zi.median()

        return med

    def _depth_mode_filter(
        self,
        Z: torch.Tensor,  # (M, H, W) depth
        weights: torch.Tensor,  # (M, H, W) float, inliers > 0
    ) -> torch.Tensor:
        """Restrict per-detection mask to depths within k MADs of the
        median (bi-modal-robust background-leak rejection).

        Per-detection loop is O(n_inliers) per mask, dominated by torch.median
        (linearithmic in n_inliers via torch internals). For typical M < 50
        detections per frame and n_inliers < 50k, runtime is <5 ms on GPU —
        scales with detection count, not scene size or time.
        """
        if not self.depth_mode_filter_enabled or weights.numel() == 0:
            return weights
        M = int(weights.shape[0])
        if M == 0:
            return weights
        out = weights.clone()
        k_mad = float(self.depth_mode_k_mad)
        min_mad = float(self.depth_mode_min_mad_m)
        for i in range(M):
            m_bool = weights[i] > 0
            n_inliers = int(m_bool.sum().item())
            if n_inliers < int(self.min_depth_points):
                continue
            zi = Z[i].masked_select(m_bool)
            med = zi.median()
            mad = (zi - med).abs().median().clamp_min(min_mad)
            lo = med - k_mad * mad
            hi = med + k_mad * mad
            depth_keep = (Z[i] >= lo) & (Z[i] <= hi)
            out[i] = out[i] * depth_keep.to(out.dtype)
        return out

    def _mahalanobis_reject(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        weights: torch.Tensor,
        means: torch.Tensor,
        cov6: torch.Tensor,
    ) -> torch.Tensor:
        if self.mahalanobis_thresh <= 0.0 or weights.numel() == 0:
            return weights
        device = weights.device
        dtype = weights.dtype
        M = weights.shape[0]
        cov = torch.empty((M, 3, 3), dtype=dtype, device=device)
        cov[:, 0, 0] = cov6[:, 0]
        cov[:, 0, 1] = cov6[:, 1]
        cov[:, 0, 2] = cov6[:, 2]
        cov[:, 1, 0] = cov6[:, 1]
        cov[:, 1, 1] = cov6[:, 3]
        cov[:, 1, 2] = cov6[:, 4]
        cov[:, 2, 0] = cov6[:, 2]
        cov[:, 2, 1] = cov6[:, 4]
        cov[:, 2, 2] = cov6[:, 5]
        cov = cov + torch.eye(3, device=device, dtype=dtype) * self._mahalanobis_eps
        inv_cov = torch.linalg.inv(cov)
        inv_cov = 0.5 * (inv_cov + inv_cov.transpose(1, 2))

        dx = X - means[:, 0].view(M, 1, 1)
        dy = Y - means[:, 1].view(M, 1, 1)
        dz = Z - means[:, 2].view(M, 1, 1)

        a = inv_cov[:, 0, 0].view(M, 1, 1)
        b = inv_cov[:, 0, 1].view(M, 1, 1)
        c = inv_cov[:, 0, 2].view(M, 1, 1)
        d = inv_cov[:, 1, 1].view(M, 1, 1)
        e = inv_cov[:, 1, 2].view(M, 1, 1)
        f = inv_cov[:, 2, 2].view(M, 1, 1)
        d2 = a * dx * dx + d * dy * dy + f * dz * dz + 2.0 * b * dx * dy + 2.0 * c * dx * dz + 2.0 * e * dy * dz

        thresh2 = float(self.mahalanobis_thresh) ** 2
        inliers = (d2 <= thresh2) & (weights > 0)
        return inliers.to(dtype=dtype)

    def _normalize_batch_inputs(
        self,
        color: torch.Tensor | Sequence[torch.Tensor],
        depth: torch.Tensor | Sequence[torch.Tensor],
        intrinsics: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], bool]:
        def _to_sequence(value: torch.Tensor | Sequence[torch.Tensor], name: str) -> tuple[list[torch.Tensor], bool]:
            if isinstance(value, torch.Tensor):
                if value.ndim >= 4:
                    return list(value.unbind(0)), False
                return [value], True
            if isinstance(value, Sequence):
                if len(value) == 0:
                    raise ValueError(f"{name} sequence cannot be empty.")
                return list(value), False
            raise TypeError(f"{name} must be a tensor or a sequence of tensors.")

        colors, colors_single = _to_sequence(color, "color")
        depths, depths_single = _to_sequence(depth, "depth")
        intrs, intrs_single = _to_sequence(intrinsics, "intrinsics")

        if not (len(colors) == len(depths) == len(intrs)):
            raise ValueError("color, depth, and intrinsics must have the same batch length.")
        return (
            colors,
            depths,
            intrs,
            bool(colors_single and depths_single and intrs_single and len(colors) == 1),
        )

    def _build_lrpc_col_lookup(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Build lookup tensors that map each prediction column to its LRPC FPN level and spatial index.
        Returns (levels, positions), each of shape (num_preds,).
        """
        if self._lrpc_num_levels == 0:
            return None, None

        levels: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for lvl, mask in enumerate(self._lrpc_masks):
            if mask is None or mask.numel() == 0:
                return None, None
            idxs = mask.nonzero(as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue
            levels.append(torch.full_like(idxs, lvl))
            positions.append(idxs)

        if not levels:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        return torch.cat(levels, dim=0), torch.cat(positions, dim=0)

    def _compute_lrpc_mask_embeddings(
        self,
        masks_lb: torch.Tensor,  # (M, H_lb, W_lb) bool
        batch_ids: torch.Tensor,  # (M,)
        col_indices: torch.Tensor,  # (M,)
        col_levels: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute mask-averaged embeddings from LRPC cls feature maps for each detection.
        """
        num_det = masks_lb.shape[0]
        if num_det == 0 or col_levels is None:
            return torch.zeros((num_det, self.feature_dim), device=self.device, dtype=torch.float32)

        embeddings = torch.zeros((num_det, self.feature_dim), device=self.device, dtype=torch.float32)
        for det_idx in range(num_det):
            col_idx = int(col_indices[det_idx].item()) if col_indices.numel() > det_idx else -1
            if col_idx < 0 or col_idx >= col_levels.numel():
                continue

            level = int(col_levels[col_idx].item())
            if level >= len(self._lrpc_cls_features):
                continue

            feat_map = self._lrpc_cls_features[level]
            if feat_map is None:
                continue

            b_idx = int(batch_ids[det_idx].item())
            if b_idx >= feat_map.shape[0]:
                continue

            mask_resized = (
                F.interpolate(
                    masks_lb[det_idx : det_idx + 1].float().unsqueeze(1),  # (1,1,H,W)
                    size=feat_map.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

            weight_sum = mask_resized.sum()
            if weight_sum.item() <= 0:
                continue

            weighted = feat_map[b_idx] * mask_resized
            embeddings[det_idx] = weighted.flatten(1).sum(dim=1) / weight_sum.clamp_min(1e-6)

        return embeddings

    def _compute_dino_mask_embeddings(
        self,
        colors: list[torch.Tensor],
        masks: torch.Tensor,  # (M, H, W) bool
        batch_ids: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        """
        Compute mask-averaged DINO features for each detection.
        """
        num_det = masks.shape[0]
        if num_det == 0:
            return torch.zeros((0, self.feature_dim), device=self.device, dtype=torch.float32)
        if self.dino_extractor is None:
            LOGGER.warning("DINO extractor not initialized; returning zero features.")
            return torch.zeros((num_det, self.feature_dim), device=self.device, dtype=torch.float32)

        grids = self.dino_extractor(colors)  # list of [H', W', C] on CPU
        if len(grids) == 0:
            return torch.zeros((num_det, self.feature_dim), device=self.device, dtype=torch.float32)

        # Stack to [B, H, W, C] and move once to target device
        try:
            grid_tensor = torch.stack(
                [g.to(self.device) if g.device != self.device else g for g in grids],
                dim=0,
            )
        except Exception:
            # Fallback if shapes mismatch; preserve behavior by returning zeros
            return torch.zeros((num_det, self.feature_dim), device=self.device, dtype=torch.float32)

        B, gH, gW, gC = grid_tensor.shape
        embeddings = torch.zeros((num_det, gC), device=self.device, dtype=torch.float32)

        batch_ids = batch_ids.to(self.device, dtype=torch.long) if batch_ids.numel() else batch_ids
        valid_det = (batch_ids >= 0) & (batch_ids < B)
        if not torch.any(valid_det):
            return embeddings

        valid_idx = torch.nonzero(valid_det, as_tuple=False).squeeze(1)
        batch_ids_valid = batch_ids[valid_idx]

        # Resize masks for valid detections in a single call: [M_valid, H', W']
        masks_valid = masks[valid_idx].to(device=self.device, dtype=torch.float32)
        masks_resized = F.interpolate(
            masks_valid.unsqueeze(1),
            size=(gH, gW),
            mode="nearest",
        ).squeeze(
            1
        )  # (M_valid, H', W')

        # Gather per-detection grids: [M_valid, H', W', C]
        grids_det = grid_tensor[batch_ids_valid]  # uses batch_ids as indices
        weights = (masks_resized > 0.5).unsqueeze(-1).to(dtype=grids_det.dtype)  # (M_valid, H', W', 1)

        weight_sum = weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)  # (M_valid, 1, 1, 1)
        pooled = (grids_det * weights).sum(dim=(1, 2), keepdim=False) / weight_sum.squeeze(2).squeeze(1)

        embeddings[valid_idx] = pooled
        return embeddings

    def _extract_box_embeddings(
        self,
        boxes_list: list[torch.Tensor],  # List of [N_i, 4] in original image coords
        orig_shapes: list[tuple[int, int]],  # List of (H, W) for each image
    ) -> torch.Tensor:
        """
        Extract visual embeddings for each bounding box from SAVPE.cv3 feature map.
        Uses adaptive pooling to extract a fixed-size embedding from each box region.
        """
        if self._savpe_cv3_features is None or self._savpe_cv3_features.numel() == 0:
            # Fallback to zeros if no features captured
            total_dets = sum(b.shape[0] for b in boxes_list)
            return torch.zeros((total_dets, self.feature_dim), device=self.device, dtype=torch.float32)

        # SAVPE cv3 output: [B, embed, H, W]
        feat_batch = self._savpe_cv3_features
        B, C, fH, fW = feat_batch.shape
        batch_size = len(boxes_list)

        if B != batch_size:
            LOGGER.warning(
                "SAVPE cv3 batch size (%d) doesn't match input batch size (%d)",
                B,
                batch_size,
            )

        all_embeddings = []
        for b_idx, (boxes, orig_shape) in enumerate(zip(boxes_list, orig_shapes)):
            if boxes.shape[0] == 0:
                all_embeddings.append(torch.empty((0, self.feature_dim), device=self.device, dtype=torch.float32))
                continue

            # Extract feature map for this batch item
            feat_map = feat_batch[b_idx] if b_idx < B else feat_batch[0]  # [C, H, W]
            orig_h, orig_w = orig_shape

            # Scale boxes from original image coords to feature map coords
            scale_h = fH / orig_h
            scale_w = fW / orig_w
            boxes_scaled = boxes.clone()
            boxes_scaled[:, [0, 2]] *= scale_w
            boxes_scaled[:, [1, 3]] *= scale_h

            # Extract features for each box using ROI pooling
            box_embeddings = []
            for box in boxes_scaled:
                x1, y1, x2, y2 = box
                x1, y1 = int(x1.clamp(0, fW - 1)), int(y1.clamp(0, fH - 1))
                x2, y2 = int(x2.clamp(x1 + 1, fW)), int(y2.clamp(y1 + 1, fH))

                # Extract and pool the region
                roi_feat = feat_map[:, y1:y2, x1:x2]  # [C, h, w]
                if roi_feat.numel() > 0:
                    pooled = F.adaptive_avg_pool2d(roi_feat.unsqueeze(0), (1, 1)).squeeze()  # [C]
                else:
                    pooled = torch.zeros(C, device=feat_map.device, dtype=feat_map.dtype)
                box_embeddings.append(pooled)

            all_embeddings.append(torch.stack(box_embeddings))

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        return torch.zeros((0, self.feature_dim), device=self.device, dtype=torch.float32)

    def _class_names_from_ids(self, class_ids: torch.Tensor) -> list[str]:
        if class_ids.numel() == 0:
            return []
        names = []
        for idx in class_ids.detach().to("cpu").tolist():
            if 0 <= idx < len(self.names):
                names.append(self.names[idx])
            else:
                names.append(str(idx))
        return names

    def _build_segmentation_outputs(
        self,
        det_per_img: Sequence[int],
        orig_shapes: Sequence[tuple[int, int]],
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        masks: torch.Tensor | Sequence[torch.Tensor],
        features: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
    ) -> list[SegmentationOutput]:
        outputs: list[SegmentationOutput] = []
        cursor = 0
        total = int(boxes.shape[0])
        device = boxes.device
        for img_idx, num_det in enumerate(det_per_img):
            if num_det == 0:
                outputs.append(
                    SegmentationOutput.empty(
                        image_shape=orig_shapes[img_idx],
                        feature_dim=self.feature_dim,
                        device=device,
                    )
                )
            else:
                span = slice(cursor, cursor + num_det)
                if isinstance(masks, torch.Tensor):
                    masks_span = masks[span]
                else:
                    masks_slice = list(masks[span])
                    if masks_slice:
                        masks_span = torch.stack(masks_slice, dim=0).to(device=device, dtype=torch.bool)
                    else:
                        h, w = orig_shapes[img_idx]
                        masks_span = torch.empty((0, h, w), dtype=torch.bool, device=device)
                outputs.append(
                    SegmentationOutput(
                        boxes_xyxy=boxes[span],
                        scores=scores[span],
                        class_ids=class_ids[span],
                        class_names=self._class_names_from_ids(class_ids[span]),
                        masks=masks_span,
                        features=features[span],
                        means=means[span],
                        covariances=covariances[span],
                    )
                )
            cursor += num_det

        if cursor != total:
            raise RuntimeError(
                f"Mismatch between packed detections ({total}) and per-image splits ({cursor}).",
            )
        if len(outputs) != len(orig_shapes):
            raise RuntimeError("Segmentation output count does not match batch size.")
        return outputs

    def _scale_masks_to_original(
        self,
        masks: torch.Tensor,  # (N, H_pad, W_pad) from ops.process_mask
        orig_shape: tuple[int, int],  # (orig_h, orig_w)
        ratio_pad: tuple[tuple[float, float], tuple[float, float]],
    ) -> torch.Tensor:
        """
        Mirror ``ultralytics.utils.ops.scale_image`` fully on the current device to avoid CPU copies.
        """
        orig_h, orig_w = int(orig_shape[0]), int(orig_shape[1])
        if masks.numel() == 0:
            return torch.zeros((0, orig_h, orig_w), dtype=torch.bool, device=self.device)

        pad = ratio_pad[1]
        top = int(pad[1])
        left = int(pad[0])
        bottom = int(masks.shape[1] - pad[1])
        right = int(masks.shape[2] - pad[0])

        masks_float = masks.to(dtype=torch.float32, device=self.device)
        cropped = masks_float[:, max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)]
        if cropped.shape[1] != orig_h or cropped.shape[2] != orig_w:
            cropped = F.interpolate(
                cropped.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return (cropped > 0.5).to(torch.bool)

    @torch.no_grad()  # inference_mode breaks YOLOE head (Conv2d expects version counters)
    def __call__(
        self,
        color: torch.Tensor | Sequence[torch.Tensor],
        depth: torch.Tensor | Sequence[torch.Tensor],
        depth_intrinsics: torch.Tensor | Sequence[torch.Tensor],
        *,
        camera_names: list[str] | None = None,
        extra_vocab: list[str] | None = None,
        extra_thres: float = 0.35,
        extra_topk: int = 1,
        offline_debug: bool = False,
    ) -> SegmentationOutput | list[SegmentationOutput]:

        # ----- Stage 1: Normalize inputs and preprocess RGB-D batch -----
        timings = {}
        if self._timing_enabled:
            t0 = time.perf_counter()
        colors, depths, depth_intrs, return_single = self._normalize_batch_inputs(color, depth, depth_intrinsics)
        batch_size = len(colors)
        self._savpe_cv3_features = None  # Clear SAVPE cv3 feature buffer
        if self._lrpc_num_levels:
            self._lrpc_cls_features = [None] * self._lrpc_num_levels
            self._lrpc_masks = [None] * self._lrpc_num_levels
        depth_maps = [self._prepare_depth(d).to(self.device, non_blocking=True) for d in depths]

        prepared = [self._preprocess(c) for c in colors]
        tensors = [item[0] for item in prepared]
        ratios = [item[1] for item in prepared]
        pads = [item[2] for item in prepared]
        orig_shapes = [item[3] for item in prepared]
        input_tensor = torch.cat(tensors, dim=0)

        if self._timing_enabled:
            t1 = time.perf_counter()
            timings["preprocess"] = t1 - t0

        # ----- Stage 2: Model inference (forward pass + prototypes) -----
        if self._timing_enabled:
            t0 = time.perf_counter()
        preds = self.model.model(input_tensor)
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            raw, aux = preds
        else:
            raw, aux = preds, None
        proto = aux[-1] if isinstance(aux, tuple) else aux
        if proto is None:
            raise RuntimeError("YOLOE segmentation head did not return mask prototypes.")
        if proto.ndim == 4:
            proto_batch = [proto[i] for i in range(proto.shape[0])]
        else:
            proto_batch = [proto]
        if len(proto_batch) == 1 and batch_size > 1:
            proto_batch = proto_batch * batch_size
        if len(proto_batch) != batch_size:
            raise RuntimeError(
                f"Mismatched prototype batch (got {len(proto_batch)}, expected {batch_size}).",
            )

        # Attach column indices to predictions so we can map detections back to LRPC feature maps
        col_levels = None
        attach_lrpc_indices = False
        if self._lrpc_num_levels and isinstance(raw, torch.Tensor):
            col_levels, _ = self._build_lrpc_col_lookup()
            if col_levels is not None and col_levels.numel() == raw.shape[2]:
                if raw.dtype != torch.float32:
                    raw = raw.float()
                idx_channel = torch.arange(raw.shape[2], device=raw.device, dtype=raw.dtype).view(1, 1, -1)
                idx_channel = idx_channel.expand(raw.shape[0], -1, -1)
                raw = torch.cat([raw, idx_channel], dim=1)
                attach_lrpc_indices = True
            else:
                LOGGER.debug(
                    "LRPC lookup unavailable (levels=%s, pred=%d); falling back to box-pooled embeddings.",
                    "None" if col_levels is None else col_levels.numel(),
                    int(raw.shape[2]),
                )

        # ----- Stage 3: Postprocessing (NMS, mask decode, geometry) -----
        head_nc = getattr(self._head, "nc", None)
        if head_nc is not None and head_nc != len(self.names):
            LOGGER.debug(
                "YOLOE head reports %d classes but vocab has %d entries.",
                head_nc,
                len(self.names),
            )

        nms_output = ops.non_max_suppression(
            raw,
            self.conf_thres,
            self.iou_thres,
            max_det=self.max_det,
            nc=head_nc or 0,
        )
        if self._timing_enabled:
            t1 = time.perf_counter()
            timings["model_inference"] = t1 - t0
            t0 = time.perf_counter()

        det_per_img: list[int] = []
        boxes_list: list[torch.Tensor] = []
        scores_list: list[torch.Tensor] = []
        class_ids_list: list[torch.Tensor] = []
        masks_ragged: list[torch.Tensor] = []
        masks_letterbox_list: list[torch.Tensor] = []
        batch_ids_chunks: list[torch.Tensor] = []
        col_indices_list: list[torch.Tensor] = []
        img_hw = input_tensor.shape[2:]
        mask_dim = getattr(self._head, "nm", 0)

        if self._timing_enabled:
            tp0 = time.perf_counter()

        for b_idx, det in enumerate(nms_output):
            if det is None or det.numel() == 0:
                det_per_img.append(0)
                # Append empty tensors for images with no detections
                boxes_list.append(torch.empty((0, 4), device=self.device, dtype=torch.float32))
                scores_list.append(torch.empty((0,), device=self.device, dtype=torch.float32))
                class_ids_list.append(torch.empty((0,), device=self.device, dtype=torch.long))
                masks_letterbox_list.append(
                    torch.empty((0, img_hw[0], img_hw[1]), device=self.device, dtype=torch.bool)
                )
                batch_ids_chunks.append(torch.empty((0,), device=self.device, dtype=torch.int64))
                col_indices_list.append(torch.empty((0,), device=self.device, dtype=torch.long))
                continue
            n_b = det.shape[0]
            det_per_img.append(n_b)

            boxes_b = det[:, :4].clone()
            scores_b = det[:, 4].clone()
            class_ids_b = det[:, 5].long()
            mask_coeffs_b_full = det[:, 6:]
            if attach_lrpc_indices and mask_dim > 0:
                mask_coeffs_b = mask_coeffs_b_full[:, :mask_dim]
                det_indices_b = mask_coeffs_b_full[:, mask_dim:].long()
            else:
                mask_coeffs_b = mask_coeffs_b_full
                det_indices_b = torch.empty((n_b,), device=self.device, dtype=torch.long)

            masks_b = ops.process_mask(
                proto_batch[b_idx],
                mask_coeffs_b,
                det[:, :4],
                img_hw,
                upsample=True,
            )
            boxes_scaled_b = boxes_b.clone()
            ops.scale_boxes(
                img_hw,
                boxes_scaled_b,
                orig_shapes[b_idx],
                ratio_pad=(ratios[b_idx], pads[b_idx]),
            )
            masks_scaled_b = self._scale_masks_to_original(
                masks_b,
                orig_shapes[b_idx],
                ratio_pad=(ratios[b_idx], pads[b_idx]),
            )

            boxes_list.append(boxes_scaled_b)
            scores_list.append(scores_b)
            class_ids_list.append(class_ids_b)
            masks_ragged.extend(masks_scaled_b.to(torch.bool).to(self.device).unbind(0))
            masks_letterbox_list.append((masks_b > 0.5).to(torch.bool).to(self.device))
            batch_ids_chunks.append(torch.full((n_b,), b_idx, device=self.device, dtype=torch.int64))
            col_indices_list.append(det_indices_b)

        M = int(sum(det_per_img))

        if M > 0:
            boxes = torch.cat(boxes_list, dim=0)
            scores = torch.cat(scores_list, dim=0)
            class_ids = torch.cat(class_ids_list, dim=0)
            masks = masks_ragged
            batch_ids = torch.cat(batch_ids_chunks, dim=0)
            col_indices = (
                torch.cat(col_indices_list, dim=0)
                if attach_lrpc_indices
                else torch.empty((0,), device=self.device, dtype=torch.long)
            )
            # Always keep letterbox-aligned masks so depth/intrinsics can be batched even when original
            # image resolutions differ (e.g., 640x480 vs 480x640).
            masks_letterbox = torch.cat(masks_letterbox_list, dim=0)
        else:
            boxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
            scores = torch.empty((0,), device=self.device, dtype=torch.float32)
            class_ids = torch.empty((0,), device=self.device, dtype=torch.long)
            if batch_size > 0:
                h0, w0 = orig_shapes[0]
            masks = []
            batch_ids = torch.empty((0,), device=self.device, dtype=torch.int64)
            col_indices = torch.empty((0,), device=self.device, dtype=torch.long)
            masks_letterbox = torch.empty((0, img_hw[0], img_hw[1]), device=self.device, dtype=torch.bool)

        # Letterbox depth maps to the same target size as the RGB model input so we can batch-stack them.
        # This keeps the entire 3D-stat computation vectorized on GPU even with mixed input resolutions.
        target_hw = (int(img_hw[0]), int(img_hw[1]))
        depth_letterbox: list[torch.Tensor] = []
        depth_intr_letterbox: list[torch.Tensor] = []
        for depth_map, depth_intr4, orig_shape in zip(depth_maps, depth_intrs, orig_shapes):
            lb_depth, gain, (pad_left, pad_top, _, _) = self._letterbox_depth(depth_map, orig_shape, target_hw)
            depth_letterbox.append(lb_depth)
            K = depth_intr4[:3, :3].to(device=self.device, dtype=lb_depth.dtype)
            fx = K[0, 0] * gain
            fy = K[1, 1] * gain
            cx = K[0, 2] * gain + float(pad_left)
            cy = K[1, 2] * gain + float(pad_top)
            K_adj = torch.zeros((3, 3), device=self.device, dtype=lb_depth.dtype)
            K_adj[0, 0] = fx
            K_adj[1, 1] = fy
            K_adj[0, 2] = cx
            K_adj[1, 2] = cy
            K_adj[2, 2] = 1.0
            depth_intr_letterbox.append(K_adj)

        depth = torch.stack(depth_letterbox, dim=0)  # (B, target_h, target_w)
        depth_intr = torch.stack(depth_intr_letterbox, dim=0)  # (B,3,3)
        orig_hw = torch.as_tensor(orig_shapes, device=self.device, dtype=torch.int32)
        ratios_t = torch.as_tensor(ratios, device=self.device, dtype=torch.float32)  # (B,2)
        pads_t = torch.as_tensor(pads, device=self.device, dtype=torch.float32)  # (B,2)

        # Per-mask 3D stats for the entire batch
        # Precompute XYZ maps per image once
        B, H, W = depth.shape
        gy, gx = self._get_pixel_grid(H, W, depth.device, depth.dtype)
        ZB = depth
        XB = (gx - depth_intr[:, 0, 2].view(B, 1, 1)) * ZB / depth_intr[:, 0, 0].view(B, 1, 1)
        YB = (gy - depth_intr[:, 1, 2].view(B, 1, 1)) * ZB / depth_intr[:, 1, 1].view(B, 1, 1)

        depth_valid_B = (ZB > 0) & torch.isfinite(ZB)
        dtype = ZB.dtype
        device = ZB.device
        masks_depth = self._erode_masks(masks_letterbox, self.mask_erosion_px)

        if self._timing_enabled:
            tp1 = time.perf_counter()
            timings["mask_decode"] = tp1 - tp0
            tp0 = time.perf_counter()

        M = int(batch_ids.numel())
        masks_eroded_ragged: list = []
        masks_inlier_ragged: list = []
        det_points_flat = torch.empty((0, 3), dtype=dtype, device=device)
        det_points_offsets = torch.zeros((M + 1,), dtype=torch.long, device=device)
        if M == 0:
            means = torch.empty((0, 3), dtype=dtype, device=device)
            cov6 = torch.empty((0, 6), dtype=dtype, device=device)
            n = torch.empty((0,), dtype=dtype, device=device)
            if offline_debug:
                n_original_pixels = torch.empty((0,), dtype=dtype, device=device)
                n_eroded_pixels = torch.empty((0,), dtype=dtype, device=device)
        else:
            batch_ids_long = batch_ids.to(torch.long)
            depth_valid = depth_valid_B[batch_ids_long]
            weights = (masks_depth & depth_valid).to(dtype)

            XB_sel = XB[batch_ids_long]
            YB_sel = YB[batch_ids_long]
            ZB_sel = ZB[batch_ids_long]

            # 1-D depth-mode filter: cut mask-background leak before
            # computing 3-D stats. See _depth_mode_filter docstring.
            weights = self._depth_mode_filter(ZB_sel, weights)

            n, means, cov6 = self._compute_weighted_stats(XB_sel, YB_sel, ZB_sel, weights)
            if self.mahalanobis_thresh > 0.0:
                weights = self._mahalanobis_reject(XB_sel, YB_sel, ZB_sel, weights, means, cov6)
                n, means, cov6 = self._compute_weighted_stats(XB_sel, YB_sel, ZB_sel, weights)

            means = self._compute_mask_medians(XB_sel, YB_sel, ZB_sel, weights, self.min_depth_points)

            # Materialize per-detection 3D point clouds (camera frame) for the
            # inliers retained after mahalanobis rejection. Stored as a single
            # flat (P, 3) tensor + (M+1,) CSR offsets so it can be transformed
            # to world frame in one batched pass downstream.
            inlier_mask = weights > 0
            if inlier_mask.any():
                det_idx_t, py_t, px_t = torch.nonzero(inlier_mask, as_tuple=True)
                pts_x = XB_sel[det_idx_t, py_t, px_t]
                pts_y = YB_sel[det_idx_t, py_t, px_t]
                pts_z = ZB_sel[det_idx_t, py_t, px_t]
                det_points_flat = torch.stack([pts_x, pts_y, pts_z], dim=1).to(dtype)
                # det_idx_t comes out sorted by (det_idx, py, px) since nonzero
                # iterates in row-major over (M, H, W). Use bincount for offsets.
                counts = torch.bincount(det_idx_t, minlength=M)
                det_points_offsets = torch.zeros((M + 1,), dtype=torch.long, device=device)
                det_points_offsets[1:] = counts.cumsum(dim=0)

            if offline_debug:
                n_original_pixels = masks_letterbox.to(dtype).sum(dim=(1, 2))
                n_eroded_pixels = masks_depth.to(dtype).sum(dim=(1, 2))
                # Scale eroded and inlier masks to original image space for debug visualization
                batch_size = len(orig_shapes)
                for b_idx in range(batch_size):
                    sel = batch_ids_long == b_idx
                    if not sel.any():
                        continue
                    indices = sel.nonzero(as_tuple=True)[0]
                    eroded_b = masks_depth[indices].to(torch.float32)
                    inlier_b = weights[indices]
                    scaled_eroded = self._scale_masks_to_original(
                        eroded_b, orig_shapes[b_idx], (ratios[b_idx], pads[b_idx])
                    )
                    scaled_inlier = self._scale_masks_to_original(
                        inlier_b, orig_shapes[b_idx], (ratios[b_idx], pads[b_idx])
                    )
                    masks_eroded_ragged.extend(scaled_eroded.to(torch.bool).unbind(0))
                    masks_inlier_ragged.extend(scaled_inlier.to(torch.bool).unbind(0))


        if self._timing_enabled:
            tp1 = time.perf_counter()
            timings["3d_stats"] = tp1 - tp0

        # Extract visual embeddings (either DINO or YOLOE features)
        if self._timing_enabled:
            tp0 = time.perf_counter()
        if self._use_dino_features:
            vis_features = self._compute_dino_mask_embeddings(colors, masks_letterbox, batch_ids)
        elif attach_lrpc_indices:
            vis_features = self._compute_lrpc_mask_embeddings(
                masks_letterbox,
                batch_ids,
                col_indices,
                col_levels,
            )
        else:
            vis_features = self._extract_box_embeddings(boxes_list, orig_shapes)
        if self._timing_enabled:
            tp1 = time.perf_counter()
            timings["feature_extraction"] = tp1 - tp0

        # ----- Stage 4: Package outputs for downstream consumers -----

        if self._timing_enabled:
            t1 = time.perf_counter()
            timings["postprocessing"] = t1 - t0

        if self._visualizer.enabled:
            if self._timing_enabled:
                t_vis0 = time.perf_counter()
            batch_outputs = self._build_segmentation_outputs(
                det_per_img=det_per_img,
                orig_shapes=orig_shapes,
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                masks=masks,
                features=vis_features,
                means=means,
                covariances=cov6,
            )
            self._visualizer.save_batch(colors, batch_outputs)

            if self._timing_enabled:
                t_vis1 = time.perf_counter()
                timings["visualization"] = t_vis1 - t_vis0

        if camera_names is None:
            camera_names = [f"cam{idx}" for idx in range(batch_size)]
        if extra_vocab is None:
            extra_vocab = []

        M = int(batch_ids.numel())
        detections: list[dict[str, object]] = []
        extra_vocab_match = torch.zeros((M,), device=self.device, dtype=torch.bool)
        if M > 0:
            colors_hwc: list[torch.Tensor | None] = []
            for color_tensor in colors:
                color_cpu = color_tensor.detach().to("cpu")
                if color_cpu.ndim == 3:
                    if color_cpu.shape[0] == 3 and color_cpu.shape[-1] != 3:
                        color_cpu = color_cpu.permute(1, 2, 0)
                elif color_cpu.ndim == 4 and color_cpu.shape[0] == 1:
                    color_cpu = color_cpu.squeeze(0)
                    if color_cpu.ndim == 3 and color_cpu.shape[0] == 3 and color_cpu.shape[-1] != 3:
                        color_cpu = color_cpu.permute(1, 2, 0)
                else:
                    color_cpu = None
                colors_hwc.append(color_cpu)

            extra_best_sim = torch.empty((M,), device=self.device, dtype=torch.float32)
            extra_best_idx = torch.full((M,), -1, device=self.device, dtype=torch.long)
            if extra_vocab and vis_features.numel() > 0:
                text_emb = self._get_extra_text_emb(extra_vocab)
                if text_emb.numel() > 0:
                    feat = F.normalize(vis_features.float(), dim=1)
                    if feat.shape[1] != text_emb.shape[1]:
                        LOGGER.warning(
                            "Extra vocab embedding dim (%d) != visual feature dim (%d); skipping overlay.",
                            int(text_emb.shape[1]),
                            int(feat.shape[1]),
                        )
                    else:
                        text_emb = text_emb.to(device=feat.device, dtype=feat.dtype)
                        sim = feat @ text_emb.t()
                        if extra_topk > 1 and sim.shape[1] > 1:
                            k = min(int(extra_topk), int(sim.shape[1]))
                            extra_best_sim, extra_best_idx = sim.topk(k=k, dim=1)
                            extra_best_sim = extra_best_sim[:, 0]
                            extra_best_idx = extra_best_idx[:, 0]
                        else:
                            extra_best_sim, extra_best_idx = sim.max(dim=1)
                        extra_vocab_match = (extra_best_idx >= 0) & (extra_best_sim >= float(extra_thres))

            base_names = self._class_names_from_ids(class_ids)
            boxes_cpu = boxes.detach().to("cpu")
            scores_cpu = scores.detach().to("cpu")
            batch_ids_cpu = batch_ids.detach().to("cpu")
            extra_best_sim_cpu = extra_best_sim.detach().to("cpu")
            extra_best_idx_cpu = extra_best_idx.detach().to("cpu")
            extra_vocab_match_cpu = extra_vocab_match.detach().to("cpu")

            for i in range(M):
                b_id = int(batch_ids_cpu[i].item())
                cam = camera_names[b_id] if 0 <= b_id < len(camera_names) else f"cam{b_id}"
                bb = boxes_cpu[i].tolist()
                cat = base_names[i] if i < len(base_names) else str(int(class_ids[i].item()))
                conf = float(scores_cpu[i].item())
                extra_match = False

                if extra_vocab:
                    s = float(extra_best_sim_cpu[i].item())
                    j = int(extra_best_idx_cpu[i].item())
                    extra_match = bool(extra_vocab_match_cpu[i].item())
                    if extra_match and 0 <= j < len(extra_vocab):
                        cat = extra_vocab[j]
                        conf = s

                detections.append({
                    "Camera Name": cam,
                    "BB": bb,
                    "Confidence": conf,
                    "Object Category": cat,
                    "Extra Vocab Match": extra_match,
                })

        out = {
            "batch_ids": batch_ids,  # (M,)
            "boxes_xyxy": boxes,  # (M,4)
            "num_pixels": n,  # (M,) inlier count after Mahalanobis
            "scores": scores,  # (M,)
            "class_ids": class_ids,  # (M,)
            "masks": masks,  # list of (H_i, W_i) bool tensors aligned to original per-camera frames
            "features": vis_features,  # (M,D)
            "means": means,  # (M,3)
            "cov6": cov6,  # (M,6)
            # Sparse per-detection 3D point clouds in camera frame; transformed
            # to world frame later by transform_segmentation_to_world. Layout is
            # CSR-flat: keys[off[i]:off[i+1]] are detection i's points.
            "det_points_flat": det_points_flat,  # (P,3) float
            "det_points_offsets": det_points_offsets,  # (M+1,) int64
            # If you truly need per-image meta, keep these as per-batch tensors for gather:
            "orig_hw": orig_hw,  # (B,2) int
            "ratios": ratios_t,  # (B,2)
            "pads": pads_t,  # (B,2)
            "detections": detections,
            "extra_vocab_matches": extra_vocab_match,
            "timings": timings,
        }
        if offline_debug:
            out["n_original_pixels"] = n_original_pixels  # (M,) mask pixels before erosion
            out["n_eroded_pixels"] = n_eroded_pixels  # (M,) mask pixels after erosion
            out["masks_eroded"] = masks_eroded_ragged
            out["masks_inlier"] = masks_inlier_ragged
        return out
