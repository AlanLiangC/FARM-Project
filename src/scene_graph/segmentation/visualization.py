from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

try:
    import supervision as sv
except ImportError:  # pragma: no cover - optional dependency
    sv = None

from .models import SegmentationOutput


def _tensor_to_pil(color: torch.Tensor) -> Image.Image:
    if color.ndim == 4 and color.shape[0] == 1:
        color = color.squeeze(0)
    if color.ndim != 3:
        raise ValueError(f"Color tensor must be CHW or HWC, got shape {tuple(color.shape)}")
    if color.shape[0] == 3 and color.shape[-1] != 3:
        chw = color
    elif color.shape[-1] == 3:
        chw = color.permute(2, 0, 1)
    else:
        raise ValueError(f"Color tensor must have 3 channels, got shape {tuple(color.shape)}")

    chw = chw.detach().to("cpu")
    if torch.is_floating_point(chw):
        if chw.max().item() <= 1.0 + 1e-3:
            chw = chw * 255.0
        chw = torch.clamp(chw, 0.0, 255.0).round().to(torch.uint8)
    else:
        chw = chw.to(torch.uint8)

    hwc = chw.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(hwc, mode="RGB")


class SegmentationVisualizer:
    """Handles optional mask visualization using supervision."""

    def __init__(self, save_dir: Path | str | None) -> None:
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self._mask_annotator: sv.MaskAnnotator | None = None
        self._counter = 0

        if self.save_dir is None:
            return
        if sv is None:
            raise ImportError(
                "supervision is required for segmentation visualization. Install it or omit --vis-segmentation."
            )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4)

    @property
    def enabled(self) -> bool:
        return self.save_dir is not None and self._mask_annotator is not None

    def _render(self, color_img: Image.Image, seg_output: SegmentationOutput) -> Image.Image:
        if not self.enabled or seg_output.num_detections == 0:
            return color_img

        detections = sv.Detections(
            xyxy=seg_output.boxes_xyxy.detach().cpu().numpy(),
            confidence=seg_output.scores.detach().cpu().numpy(),
            class_id=seg_output.class_ids.detach().cpu().numpy(),
            mask=seg_output.masks.detach().cpu().numpy().astype(bool),
        )
        detections.data["class_name"] = list(seg_output.class_names)
        annotated = self._mask_annotator.annotate(scene=color_img.copy(), detections=detections)
        return Image.fromarray(np.asarray(annotated))

    def _tile_protos(self, proto: torch.Tensor | np.ndarray, normalize: bool = True) -> Image.Image:
        if isinstance(proto, torch.Tensor):
            proto = proto.detach().cpu().numpy()
        assert proto.ndim == 3, f"Proto must be (C,H,W), got {proto.shape}"
        C, H, W = proto.shape
        p = proto.copy()
        if normalize:
            for i in range(C):
                ch = p[i]
                mn, mx = ch.min(), ch.max()
                if mx > mn:
                    p[i] = (ch - mn) / (mx - mn)
                else:
                    p[i] = np.zeros_like(ch)
            p = (p * 255.0).clip(0, 255).astype(np.uint8)
        else:
            mn, mx = p.min(), p.max()
            p = ((p - mn) / (mx - mn + 1e-12) * 255.0).clip(0, 255).astype(np.uint8)

        cols = int(np.ceil(np.sqrt(C)))
        rows = int(np.ceil(C / cols))
        canvas = np.zeros((rows * H, cols * W), dtype=np.uint8)
        for i in range(C):
            r = i // cols
            c = i % cols
            canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = p[i]
        return Image.fromarray(canvas, mode="L")

    def _save_single_detection(
        self, color_img: Image.Image, seg_output: SegmentationOutput, det_idx: int, frame_idx: int
    ):
        xyxy = seg_output.boxes_xyxy[det_idx : det_idx + 1].detach().cpu().numpy()
        conf = seg_output.scores[det_idx : det_idx + 1].detach().cpu().numpy()
        cls = seg_output.class_ids[det_idx : det_idx + 1].detach().cpu().numpy()
        mask = seg_output.masks[det_idx].detach().cpu().numpy().astype(bool)[None, ...]
        det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, mask=mask)
        canvas = self._mask_annotator.annotate(scene=color_img.copy(), detections=det)
        out_dir = self.save_dir / "detections"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"frame_{frame_idx:06d}_det_{det_idx:03d}.jpg"
        if isinstance(canvas, Image.Image):
            canvas.save(fname, quality=95)
        else:
            Image.fromarray(np.asarray(canvas)).save(fname, quality=95)

    def save_batch(
        self,
        colors: Sequence[torch.Tensor],
        outputs: Sequence[SegmentationOutput],
        protos: Optional[Sequence[torch.Tensor | np.ndarray]] = None,
        save_detections: bool = False,
        save_protos: bool = False,
    ) -> None:
        if not self.enabled:
            return

        for idx_in_batch, (color_tensor, seg_output) in enumerate(zip(colors, outputs)):
            pil_img = _tensor_to_pil(color_tensor)
            overlay = self._render(pil_img, seg_output)
            frame_idx = self._counter
            filename = self.save_dir / f"frame_{frame_idx:06d}.jpg"
            overlay.save(filename, quality=95)

            if save_detections and seg_output.num_detections > 0:
                for d in range(seg_output.num_detections):
                    self._save_single_detection(pil_img, seg_output, d, frame_idx)
            if save_protos and protos is not None and idx_in_batch < len(protos) and protos[idx_in_batch] is not None:
                try:
                    img = self._tile_protos(protos[idx_in_batch])
                    img.save(self.save_dir / f"frame_{frame_idx:06d}_protos_tiled.jpg", quality=95)
                except Exception:
                    pass

            self._counter += 1
