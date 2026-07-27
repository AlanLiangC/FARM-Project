from __future__ import annotations

import math
import os
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.modules.utils as nn_utils
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# Keep gradients disabled globally for inference
torch.set_grad_enabled(False)

# Favor fast CUDA kernels when available
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    os.environ.setdefault("PYTORCH_USE_FLASH_ATTENTION", "1")
except Exception:
    pass

# DINO merge backbone. When no explicit ``weights_path`` is given, ``__init__``
# calls ``resolve_dino_backbone()``, which **auto-prefers the gated ViT-S+/16
# (``dinov3-vits16plus``)** — the paper backbone with tighter, more stable
# merging across scenes and imaging conditions — whenever a local copy is
# present, and otherwise falls back to the non-gated ViT-S/16
# (``dinov3-vits16``, checked in by bootstrap_models.sh) so a fresh clone still
# loads fully offline. ``DEFAULT_MODEL`` is the offline-safe fallback id used
# only when neither local dir exists.
DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _to_pil(img: Union[np.ndarray, torch.Tensor, Image.Image]) -> Image.Image:
    """Accept HWC/CHW torch or numpy arrays and return a PIL image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        x = img
        if x.ndim == 3 and x.shape[0] in (1, 3):  # CHW -> HWC
            x = x.permute(1, 2, 0).contiguous()
        img = x.detach().cpu().to(torch.uint8).numpy()
    if isinstance(img, np.ndarray):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 image")
        return Image.fromarray(img)
    raise TypeError(f"Unsupported image type: {type(img)}")


class DINOFeaturesExtractor:
    """
    Lightweight DINOv3 token extractor backed by Hugging Face transformers.

    Given an RGB image (or list), returns a list of per-image token grids
    shaped [H', W', D] with CLS/register tokens removed and L2-normalized.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        weights_path: str | None = None,
        load_size: int = 518,
        stride: int | None = None,
        fp16: bool = True,
        channels_last: bool = True,
        facet: str = "token",
        device: str | torch.device | None = None,
        **_,  # ignore extra kwargs to stay permissive
    ) -> None:
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.channels_last = bool(channels_last)
        self.load_size = int(load_size)
        self.fp16 = bool(fp16)
        self.facet = str(facet or "token").strip().lower()
        self._backend = "hf"
        self._hub_model_type = str(model or "")
        self._hub_stride = stride
        self._hub_feats: list[torch.Tensor] = []
        self._hub_hook_handlers: list = []

        if self._hub_model_type.startswith("dinov2_"):
            self._init_torchhub_dinov2(self._hub_model_type, stride=stride)
            return

        # If the caller did not pin an explicit weights directory, resolve the
        # merge backbone automatically: prefer the gated ViT-S+/16 when present,
        # else fall back to the checked-in non-gated ViT-S/16 (offline-safe).
        if not weights_path:
            try:
                from scene_graph.runtime_paths import resolve_dino_backbone

                resolved_model, resolved_weights = resolve_dino_backbone()
                weights_path = resolved_weights
                # Adopt the resolved model id only when the caller left the default.
                if model == DEFAULT_MODEL and resolved_model:
                    model = resolved_model
                    self._hub_model_type = str(model)
            except Exception:
                pass

        model_src = weights_path if (weights_path and os.path.exists(weights_path)) else model
        cache_dir = weights_path if weights_path else None
        local_only = bool(weights_path)

        self.processor = AutoImageProcessor.from_pretrained(
            model_src,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
        self.model = (
            AutoModel.from_pretrained(
                model_src,
                trust_remote_code=True,
                cache_dir=cache_dir,
                local_files_only=local_only,
            )
            .eval()
            .to(self.device)
        )
        if self.channels_last and self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)

        cfg = getattr(self.model, "config", object())
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        cfg_hidden = getattr(cfg, "hidden_size", None)
        cfg_hidden_sizes = getattr(cfg, "hidden_sizes", None)
        if cfg_hidden and int(cfg_hidden) > 0:
            self.hidden_size = int(cfg_hidden)
        elif cfg_hidden_sizes:
            try:
                self.hidden_size = int(cfg_hidden_sizes[-1])
            except Exception:
                self.hidden_size = None
        else:
            self.hidden_size = None

    def _init_torchhub_dinov2(self, model_type: str, *, stride: int | None = None) -> None:
        self._backend = "torchhub_dinov2"
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.model.eval().to(self.device)
        if self.channels_last and self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)

        backbone = getattr(self.model, "backbone", self.model)
        self.num_register_tokens = int(getattr(backbone, "num_register_tokens", 0))
        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is None:
            raise RuntimeError(f"DINOv2 model {model_type!r} has no patch_embed.")
        patch_size_raw = getattr(patch_embed, "patch_size", 14)
        if isinstance(patch_size_raw, (tuple, list)):
            self._hub_patch_size = int(patch_size_raw[0])
        else:
            self._hub_patch_size = int(patch_size_raw)
        if stride is not None and int(stride) > 0:
            self._patch_torchhub_vit_stride(int(stride))
        stride_raw = getattr(patch_embed.proj, "stride", (self._hub_patch_size, self._hub_patch_size))
        self._hub_stride_pair = nn_utils._pair(stride_raw)
        cfg = getattr(backbone, "embed_dim", None)
        if cfg is None:
            cfg = getattr(getattr(backbone, "norm", None), "normalized_shape", [None])[0]
        self.hidden_size = int(cfg) if cfg is not None else None
        if self.hidden_size is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, max(self.load_size, self._hub_patch_size), max(self.load_size, self._hub_patch_size), device=self.device)
                if self.device.type == "cuda" and self.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feat = self._forward_torchhub_dinov2(dummy)
                else:
                    feat = self._forward_torchhub_dinov2(dummy)
                self.hidden_size = int(feat.shape[-1])

    def _patch_torchhub_vit_stride(self, stride: int) -> None:
        backbone = getattr(self.model, "backbone", self.model)
        patch_embed = backbone.patch_embed
        patch_size = self._hub_patch_size
        stride_pair = nn_utils._pair(stride)
        if all(int(x) == int(patch_size) for x in stride_pair):
            return
        if not all((patch_size // int(x)) * int(x) == patch_size for x in stride_pair):
            raise ValueError(f"DINOv2 stride {stride_pair} must divide patch size {patch_size}.")
        patch_embed.proj.stride = stride_pair

        def interpolate_pos_encoding(module, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            pos_embed = module.pos_embed
            npatch = x.shape[1] - 1
            n_ref = pos_embed.shape[1] - 1
            if npatch == n_ref and w == h:
                return pos_embed
            class_pos_embed = pos_embed[:, 0]
            patch_pos_embed = pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = 1 + (w - patch_size) // stride_pair[1]
            h0 = 1 + (h - patch_size) // stride_pair[0]
            w0_f, h0_f = w0 + 0.1, h0 + 0.1
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(n_ref)), int(math.sqrt(n_ref)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0_f / math.sqrt(n_ref), h0_f / math.sqrt(n_ref)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        import types

        backbone.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, backbone)

    def _torchhub_preprocess_batch(self, images: Sequence[Union[np.ndarray, torch.Tensor, Image.Image]]) -> torch.Tensor:
        pil_batch = [_to_pil(im) for im in images]
        out = []
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for pil_image in pil_batch:
            if self.load_size > 0:
                pil_image = transforms.Resize(self.load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
                width, height = pil_image.size
                patch = int(self._hub_patch_size)
                if width % patch != 0:
                    width += patch - width % patch
                if height % patch != 0:
                    height += patch - height % patch
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            prep = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
            out.append(prep(pil_image))
        x = torch.stack(out, dim=0)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        if self.device.type == "cuda":
            x = x.pin_memory()
        return x.to(self.device, non_blocking=True)

    def _torchhub_hook(self, facet: str):
        if facet == "token":
            def hook(_module, _inputs, output):
                self._hub_feats.append(output)
            return hook
        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise ValueError(f"Unsupported DINO facet {facet!r}. Use token/key/query/value.")

        def hook(module, inputs, _output):
            x = inputs[0]
            bsz, ntok, channels = x.shape
            qkv = module.qkv(x).reshape(
                bsz, ntok, 3, module.num_heads, channels // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            self._hub_feats.append(qkv[facet_idx])

        return hook

    def _forward_torchhub_dinov2(self, batch: torch.Tensor) -> torch.Tensor:
        facet = self.facet if self.facet in {"token", "key", "query", "value"} else "token"
        backbone = getattr(self.model, "backbone", self.model)
        blocks = backbone.blocks
        self._hub_feats = []
        layer = len(blocks) - 1
        target = blocks[layer] if facet == "token" else blocks[layer].attn
        handle = target.register_forward_hook(self._torchhub_hook(facet))
        try:
            _ = self.model(batch)
        finally:
            handle.remove()
        if not self._hub_feats:
            raise RuntimeError("DINOv2 hook produced no features.")
        x = self._hub_feats[0]
        if facet == "token":
            # B x T x C -> B x 1 x T x C
            x = x.unsqueeze(1)
        if "reg" in self._hub_model_type:
            cls = x[:, :, 0, :].unsqueeze(2)
            spatial = x[:, :, (1 + self.num_register_tokens):, :]
            x = torch.cat([cls, spatial], dim=2)
        # Remove CLS and flatten heads when needed: B x T x C.
        x = x[:, :, 1:, :]
        x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)
        return x

    @torch.no_grad()
    def _call_torchhub_dinov2(self, images: Sequence[Union[np.ndarray, torch.Tensor, Image.Image]]) -> List[torch.Tensor]:
        batch = self._torchhub_preprocess_batch(images)
        if self.device.type == "cuda" and self.fp16:
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext

            amp_ctx = nullcontext()
        with torch.inference_mode(), amp_ctx:
            x = self._forward_torchhub_dinov2(batch).to(dtype=torch.float32)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        tokens = int(x.shape[1])
        h_img, w_img = batch.shape[-2:]
        stride_h, stride_w = self._hub_stride_pair
        grid_h = 1 + (int(h_img) - int(self._hub_patch_size)) // int(stride_h)
        grid_w = 1 + (int(w_img) - int(self._hub_patch_size)) // int(stride_w)
        if grid_h * grid_w != tokens:
            grid_h, grid_w = self._infer_grid(tokens, (h_img, w_img))
        return [x[b].reshape(grid_h, grid_w, x.shape[-1]).contiguous() for b in range(x.shape[0])]

    # --------- preprocessing ----------
    def _preprocess_batch(self, images: Sequence[Union[np.ndarray, torch.Tensor, Image.Image]]) -> dict:
        pil_batch = [_to_pil(im) for im in images]
        inputs = self.processor(
            images=pil_batch,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.load_size, "width": self.load_size},
            do_center_crop=False,
        )
        x = inputs["pixel_values"]  # Bx3xH'xW'
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        if self.device.type == "cuda":
            x = x.pin_memory()
        inputs["pixel_values"] = x.to(self.device, non_blocking=True)
        return inputs

    # --------- forward ----------
    @torch.no_grad()
    def __call__(self, images: Union[np.ndarray, torch.Tensor, Image.Image, Sequence]) -> List[torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]

        if self._backend == "torchhub_dinov2":
            return self._call_torchhub_dinov2(images)

        inputs = self._preprocess_batch(images)
        B = inputs["pixel_values"].shape[0]

        # Autocast: fp16 if requested; otherwise fp32
        if self.device.type == "cuda" and self.fp16:
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext

            amp_ctx = nullcontext()

        with torch.inference_mode(), amp_ctx:
            out = self.model(**inputs)  # out.last_hidden_state: (B, 1+T(+regs), D)
            x = out.last_hidden_state

        hidden = x.shape[-1]
        if self.hidden_size is None or self.hidden_size <= 0:
            self.hidden_size = int(hidden)

        # Remove CLS + register tokens (keep spatial tokens only)
        start = 1 + self.num_register_tokens
        if x.shape[1] <= start:
            raise ValueError("DINO output does not contain spatial tokens.")
        x = x[:, start:, :]  # (B, T, D)

        # L2-normalize per token for cosine stability
        x = torch.nn.functional.normalize(x, p=2, dim=-1)

        tokens = x.shape[1]
        ph, pw = self._infer_grid(tokens, inputs["pixel_values"].shape[-2:])
        grids: List[torch.Tensor] = [x[b].reshape(ph, pw, hidden).contiguous() for b in range(B)]
        return grids

    # Compatibility helper
    def extract_batch(self, images: Sequence[Union[np.ndarray, torch.Tensor, Image.Image]]) -> List[torch.Tensor]:
        return self.__call__(images)

    def _infer_grid(self, tokens: int, hw: tuple[int, int]) -> tuple[int, int]:
        """
        Infer a reasonable (H, W) grid for the spatial tokens given:
          - total token count
          - resized image height/width
        Falls back to the factor pair whose aspect ratio is closest to the image aspect.
        """
        if tokens <= 0:
            raise ValueError("Token count must be positive.")
        h_img, w_img = hw
        aspect = float(h_img) / float(w_img) if w_img else 1.0

        # Search over divisors for best aspect match (consider both orientations)
        best_h, best_w = 1, tokens
        best_err = float("inf")
        # Limit search to reasonable divisors
        for h in range(1, int(math.sqrt(tokens)) + 1):
            if tokens % h != 0:
                continue
            w = tokens // h
            ratio_hw = h / w if w else float("inf")
            ratio_wh = w / h if h else float("inf")
            err_hw = abs(ratio_hw - aspect)
            err_wh = abs(ratio_wh - aspect)
            if err_hw < best_err:
                best_err = err_hw
                best_h, best_w = h, w
            if err_wh < best_err:
                best_err = err_wh
                best_h, best_w = w, h

        return best_h, best_w
