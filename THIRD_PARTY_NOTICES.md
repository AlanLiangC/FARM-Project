# Third-party model notices

This repository's own code is licensed under AGPL-3.0-or-later (see
`LICENSE`). Pretrained model weights are fetched at setup time by
`bootstrap_models.sh` from their official public sources, except the DINOv3
ViT-S/16 backbone, which is redistributed in-repo together with its license.
Each model is governed by **its own license**, separate from this
repository's license. Review the terms below before using this project,
especially for commercial use.

| Model | Provider | License | Where it comes from |
|---|---|---|---|
| YOLOE (segmentation) | Ultralytics / THU-MIG | **AGPL-3.0** (commercial license available from Ultralytics) | Code: `third_party/yoloe` submodule (full text at `third_party/yoloe/LICENSE`); weights: [hf.co/jameslahm/yoloe](https://huggingface.co/jameslahm/yoloe) |
| MobileCLIP | Apple | Apple's redistribution license (permissive with attribution/trademark conditions, "AS IS") | Apple's CDN (official release); full text at `third_party/yoloe/third_party/ml-mobileclip/LICENSE_weights_data` |
| DINOv3 ViT-S/16 | Meta | **DINOv3 License** (custom agreement — using or distributing the materials constitutes acceptance) | Redistributed in this repository at `models/dinov3-vits16/`, with the agreement at `models/dinov3-vits16/LICENSE.md` as the license requires |
| DINOv3 ViT-S+/16 (optional) | Meta | DINOv3 License (gated) | User-downloaded from [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) after accepting the license |
| SigLIP2 | Google | See the model card | Downloaded on demand via `scripts/download_siglip2.py` from Hugging Face (`google/siglip2-large-patch16-256`) — not stored in this repo |

This project depends on `ultralytics` (AGPL-3.0) at the code level, not just
by downloading YOLOE weights — that's why the repository itself is licensed
AGPL-3.0-or-later rather than a more permissive license.

If you swap in a different segmentation or embedding backend, re-check which
of the above still applies to your deployment.
