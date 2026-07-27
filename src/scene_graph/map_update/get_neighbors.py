import os

import torch
import torch.nn.functional as F


# Hellinger matching kernel bandwidth (m²). Added as ε·I to every covariance
# entering the Hellinger calculation. Has two roles:
#   1. Numerical SPD ridge for the Cholesky inside _hellinger_distance.
#   2. *Matching kernel bandwidth* — defines the centroid drift scale at
#      which two same-object views are still considered "spatially the same".
# When both per-view covs are smaller than ε (typical for synthetic-rendered
# scenes with perfect depth, e.g. HM3D), Hellinger reduces to an isotropic-
# distance check and ε alone determines the admissible drift:
#     d_max ≈ √(8 ε · ln(1 / (1 − H²_thresh)))
# At ε=1e-2 m² (~10 cm σ) with H²_thresh=0.8, d_max ≈ 36 cm — the right scale
# for "different views of the same object". When per-view covs are larger (real
# RGBD, outdoor LIDAR), they dominate and ε contributes nothing. This makes
# the threshold robust across indoor/synthetic/outdoor without per-dataset tuning.
DEFAULT_HELLINGER_MATCH_FLOOR = float(
    os.environ.get("SCENE_GRAPH_HELLINGER_MATCH_FLOOR", "1e-2")
)


def _cov6_to_matrix(cov6):
    """
    Convert packed 6D covariance to full 3x3 symmetric matrices.

    Args:
        cov6: (B, 6) with format [s00, s01, s02, s11, s12, s22]

    Returns:
        S: (B, 3, 3) symmetric matrices
    """
    B = cov6.shape[0]
    S = torch.zeros(B, 3, 3, device=cov6.device, dtype=cov6.dtype)
    S[:, 0, 0] = cov6[:, 0]  # s00
    S[:, 0, 1] = S[:, 1, 0] = cov6[:, 1]  # s01
    S[:, 0, 2] = S[:, 2, 0] = cov6[:, 2]  # s02
    S[:, 1, 1] = cov6[:, 3]  # s11
    S[:, 1, 2] = S[:, 2, 1] = cov6[:, 4]  # s12
    S[:, 2, 2] = cov6[:, 5]  # s22
    return S


def _symmetrize_and_eig_floor(S: torch.Tensor, floor: float) -> torch.Tensor:
    """Symmetrize S and *conditionally* shift each entry so λ_min ≥ floor.

    For each (..., 3, 3) covariance we add ``max(0, floor − λ_min) · I``.
    Well-conditioned covs (λ_min ≥ floor) get only the symmetrize pass —
    no inflation. Ill-conditioned covs are shifted by the minimum amount
    needed to satisfy the floor.

    This is the *adaptive* matching kernel — for each Gaussian, the kernel
    bandwidth becomes max(natural cov, floor). Render-tight HM3D covs get
    bumped to the floor; real-depth ScanNet covs and large LIDAR GrandTour
    covs pass through unchanged. Same call site, same code, different
    behaviour automatically based on each object's own physics.
    """
    if S is None or not isinstance(S, torch.Tensor) or S.numel() == 0:
        return S
    S = 0.5 * (S + S.transpose(-1, -2))
    if floor <= 0.0:
        return S
    try:
        w_min = torch.linalg.eigvalsh(S)[..., 0]
    except Exception:
        # If eigvalsh fails (rare), fall back to unconditional ridge.
        eye = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).expand_as(S)
        return S + float(floor) * eye
    shift = (float(floor) - w_min).clamp(min=0.0)
    if not bool((shift > 0).any()):
        return S
    eye = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype).expand_as(S)
    return S + shift[..., None, None] * eye


def _hellinger_distance(mu1, S1, mu2, S2, eps=DEFAULT_HELLINGER_MATCH_FLOOR):
    """
    Compute Hellinger^2 distance between Gaussian distributions (batched).

    Args:
        mu1, mu2: (K, 3) means
        S1, S2: (K, 3, 3) covariances
        eps: matching kernel floor (m²). Each S gets a *conditional* shift
            ``max(0, eps − λ_min(S)) · I`` so the kernel bandwidth becomes
            ``max(natural cov scale, eps)`` per Gaussian. Tight covs (HM3D
            rendered) get bumped to ``eps``; large covs (GrandTour LIDAR)
            pass through unchanged. See ``_symmetrize_and_eig_floor``.

    Returns:
        H2: (K,) Hellinger^2 distances
    """
    K = S1.shape[0]

    # Per-Gaussian conditional eigenvalue floor (was: unconditional ridge).
    S1 = _symmetrize_and_eig_floor(S1, eps)
    S2 = _symmetrize_and_eig_floor(S2, eps)

    # Average covariance
    Sigma = 0.5 * (S1 + S2)

    # Default to maximum distance; we will overwrite entries
    # where the Cholesky factorization succeeds.
    H2 = torch.ones(K, device=S1.device, dtype=S1.dtype)

    # Use cholesky_ex to avoid raising on non-PD inputs when available
    if hasattr(torch.linalg, "cholesky_ex"):
        L1, info1 = torch.linalg.cholesky_ex(S1)
        L2, info2 = torch.linalg.cholesky_ex(S2)
        L, infoSigma = torch.linalg.cholesky_ex(Sigma)
        valid = (info1 == 0) & (info2 == 0) & (infoSigma == 0)
    else:
        try:
            L1 = torch.linalg.cholesky(S1)
            L2 = torch.linalg.cholesky(S2)
            L = torch.linalg.cholesky(Sigma)
            valid = torch.ones(K, dtype=torch.bool, device=S1.device)
        except Exception:
            # If Cholesky fails for the whole batch, keep H2=1
            return H2

    if not valid.any():
        return H2

    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

    L1_v = L1[idx]
    L2_v = L2[idx]
    L_v = L[idx]
    mu1_v = mu1[idx]
    mu2_v = mu2[idx]

    # Log determinants: log|S| = 2*sum(log(diag(L)))
    logdet_S1 = 2 * torch.log(torch.diagonal(L1_v, dim1=-2, dim2=-1)).sum(-1)
    logdet_S2 = 2 * torch.log(torch.diagonal(L2_v, dim1=-2, dim2=-1)).sum(-1)
    logdet_Sigma = 2 * torch.log(torch.diagonal(L_v, dim1=-2, dim2=-1)).sum(-1)

    # Mahalanobis term: (mu1-mu2)^T Sigma^{-1} (mu1-mu2)
    delta = (mu1_v - mu2_v).unsqueeze(-1)  # (K_valid, 3, 1)
    y = torch.cholesky_solve(delta, L_v)  # Sigma^{-1} @ delta
    quad = (delta.squeeze(-1) * y.squeeze(-1)).sum(-1)

    # Bhattacharyya coefficient & Hellinger distance
    logBC = 0.25 * (logdet_S1 + logdet_S2) - 0.5 * logdet_Sigma - 0.125 * quad
    BC = torch.exp(logBC)
    H2_valid = 1.0 - BC

    # Fill in distances for valid pairs; invalid ones remain at 1.0
    H2[idx] = H2_valid

    return H2


def _regroup_by_detection(det_ids, obj_ids, num_detections):
    """
    Regroup flat (detection_id, object_id) pairs into per-detection lists.

    Args:
        det_ids: (K,) detection indices
        obj_ids: (K,) object indices
        num_detections: M, total number of detections

    Returns:
        neighbors: list of M tensors, neighbors[i] = object indices for detection i
    """
    neighbors = [torch.empty(0, dtype=torch.long, device=det_ids.device) for _ in range(num_detections)]

    if det_ids.numel() == 0:
        return neighbors

    # Sort by detection id
    srt = torch.argsort(det_ids)
    det_sorted = det_ids[srt]
    obj_sorted = obj_ids[srt]

    # Count neighbors per detection
    counts = torch.bincount(det_sorted, minlength=num_detections)

    # Compute start/end indices
    csum = counts.cumsum(0)
    start = torch.zeros_like(csum)
    start[1:] = csum[:-1]

    # Extract slices
    for d in range(num_detections):
        if counts[d] > 0:
            neighbors[d] = obj_sorted[start[d] : csum[d]]

    return neighbors


@torch.no_grad()
def get_neighbors(
    detections_dict,
    db_dict,
    feature_sim_thresh=0.5,
    hellinger_thresh=0.8,
    active_mask=None,
    k=0,
    eps_cov=DEFAULT_HELLINGER_MATCH_FLOOR,
    use_class_gate: bool = False,
    return_diagnostics: bool = False,
):
    """
    Find neighbors for detections using multi-stage filtering:
      1. Class compatibility (with wildcard -1)
      2. Feature similarity (cosine > threshold)
      3. Hellinger distance (spatial < threshold)
      4. Optional: top-k selection

    Args:
        detections_dict: dict with keys:
            - features: (M, D) feature vectors
            - class_ids: (M,) class labels (-1 = wildcard)
            - means: (M, 3) Gaussian means
            - cov6: (M, 6) covariances [s00, s01, s02, s11, s12, s22]

        db_dict: same structure with N database objects

        feature_sim_thresh: min cosine similarity (default 0.5)
        hellinger_thresh: max Hellinger distance (default 0.85)
        active_mask: (M,) bool, only active detections get neighbors
        k: if > 0, also return top-k in fixed array
        eps_cov: epsilon for SPD enforcement
        return_diagnostics: if True, also return a small dict with vectorized
            funnel counters (cheap reductions over tensors the function
            already builds) so callers can disambiguate "no candidate at
            all" vs "had candidates, lost to Hellinger".

    Returns:
        neighbors: list of M tensors with neighbor indices (variable length)
        k_neighbors: (M, k) tensor with top-k indices, -1=padding (None if k<=0)
        diagnostics: only returned when ``return_diagnostics=True``::
            {
              "n_detections": int,                     # M
              "n_state_objects": int,                  # N
              "n_active_detections": int,              # M with active_mask=True and SPD
              "n_with_class_feat_candidates": int,     # passed feat-sim + class + active
              "n_with_hellinger_match": int,           # also passed Hellinger
            }
    """
    device = detections_dict["features"].device
    M = detections_dict["features"].shape[0]
    N = db_dict["features"].shape[0] if db_dict["features"] is not None else 0

    # Empty database case
    if N == 0:
        empty_neighbors = [torch.empty(0, dtype=torch.long, device=device) for _ in range(M)]
        empty_topk = None if k <= 0 else torch.full((M, k), -1, dtype=torch.long, device=device)
        if return_diagnostics:
            diag = {
                "n_detections": int(M),
                "n_state_objects": int(N),
                "n_active_detections": 0,
                "n_with_class_feat_candidates": 0,
                "n_with_hellinger_match": 0,
            }
            return empty_neighbors, empty_topk, diag
        return empty_neighbors, empty_topk

    # Default: all active
    if active_mask is None:
        active_mask = torch.ones(M, dtype=torch.bool, device=device)
    else:
        active_mask = active_mask.to(device=device, dtype=torch.bool)

    # Extract data
    det_feats = detections_dict["features"]  # (M, D)
    det_means = detections_dict["means"]  # (M, 3)
    det_cov6 = detections_dict["cov6"]  # (M, 6)

    db_feats = db_dict["features"]  # (N, D)
    db_means = db_dict["means"]  # (N, 3)
    db_cov6 = db_dict["cov6"]  # (N, 6)
    db_active = db_dict.get("active")
    if db_active is None:
        db_active = torch.ones(N, dtype=torch.bool, device=device)
    else:
        db_active = db_active.to(device=device, dtype=torch.bool)

    # Convert covariances
    det_cov = _cov6_to_matrix(det_cov6)  # (M, 3, 3)
    db_cov = _cov6_to_matrix(db_cov6)  # (N, 3, 3)

    # Filter out detections/objects with invalid or non-positive covariance diagonals
    det_finite = torch.isfinite(det_cov).all(dim=(1, 2)) & torch.isfinite(det_means).all(dim=1)
    db_finite = torch.isfinite(db_cov).all(dim=(1, 2)) & torch.isfinite(db_means).all(dim=1)
    det_posdiag = (torch.diagonal(det_cov, dim1=-2, dim2=-1) > 0).all(dim=1)
    db_posdiag = (torch.diagonal(db_cov, dim1=-2, dim2=-1) > 0).all(dim=1)
    det_valid = det_finite & det_posdiag
    db_valid = db_finite & db_posdiag

    # Normalize features
    det_feats = F.normalize(det_feats, p=2, dim=1, eps=1e-12)
    db_feats = F.normalize(db_feats, p=2, dim=1, eps=1e-12)

    # === Stage 1: Optional class compatibility ===
    class_ok = None
    if use_class_gate:
        det_class_ids = detections_dict.get("class_ids")
        db_class_ids = db_dict.get("class_ids")
        if isinstance(det_class_ids, torch.Tensor) and isinstance(db_class_ids, torch.Tensor):
            det_class_ids = det_class_ids.to(device=device, dtype=torch.long).view(-1)
            db_class_ids = db_class_ids.to(device=device, dtype=torch.long).view(-1)
            if det_class_ids.numel() == M and db_class_ids.numel() == N:
                class_ok = (
                    (det_class_ids[:, None] < 0)
                    | (db_class_ids[None, :] < 0)
                    | (det_class_ids[:, None] == db_class_ids[None, :])
                )

    # === Stage 2: Feature similarity ===
    feat_sim = torch.mm(det_feats, db_feats.t())  # (M, N)
    candidates = feat_sim > feature_sim_thresh  # (M, N)
    if class_ok is not None:
        candidates = candidates & class_ok
    candidates = candidates & det_valid.unsqueeze(1) & db_valid.unsqueeze(0) & db_active.unsqueeze(0)
    if active_mask is not None:
        candidates = candidates & active_mask.unsqueeze(1)

    # Vectorized funnel diagnostics (cheap — single reductions over tensors
    # already on the device). Only materialized when the caller asks.
    n_active_dets = int((active_mask & det_valid).sum().item()) if return_diagnostics else 0
    n_with_class_feat_candidates = int(candidates.any(dim=1).sum().item()) if return_diagnostics else 0

    # Get (det_id, obj_id) pairs
    pairs = torch.nonzero(candidates, as_tuple=False)  # (K, 2)

    if pairs.numel() == 0:
        empty_neighbors = [torch.empty(0, dtype=torch.long, device=device) for _ in range(M)]
        empty_topk = None if k <= 0 else torch.full((M, k), -1, dtype=torch.long, device=device)
        if return_diagnostics:
            diag = {
                "n_detections": int(M),
                "n_state_objects": int(N),
                "n_active_detections": n_active_dets,
                "n_with_class_feat_candidates": n_with_class_feat_candidates,
                "n_with_hellinger_match": 0,
            }
            return empty_neighbors, empty_topk, diag
        return empty_neighbors, empty_topk

    det_ids = pairs[:, 0]  # (K,)
    obj_ids = pairs[:, 1]  # (K,)

    # === Stage 3 + 4 Combined: Hellinger distance, filtering, and grouping ===
    # Gather aligned pairs
    mu_det = det_means[det_ids]  # (K, 3)
    cov_det = det_cov[det_ids]  # (K, 3, 3)
    mu_db = db_means[obj_ids]  # (K, 3)
    cov_db = db_cov[obj_ids]  # (K, 3, 3)

    # Compute distances
    with torch.amp.autocast("cuda", enabled=False):  # forces float32 for numerical stability
        H2 = _hellinger_distance(mu_det.float(), cov_det.float(), mu_db.float(), cov_db.float(), eps_cov)

    # Sort by detection ID once
    sort_idx = torch.argsort(det_ids)
    det_sorted = det_ids[sort_idx]
    obj_sorted = obj_ids[sort_idx]
    H2_sorted = H2[sort_idx]

    # Count candidates per detection
    counts = torch.bincount(det_sorted, minlength=M)
    end_indices = counts.cumsum(0)
    start_indices = torch.zeros_like(end_indices)
    start_indices[1:] = end_indices[:-1]

    # Initialize outputs
    neighbors = [torch.empty(0, dtype=torch.long, device=device) for _ in range(M)]
    k_neighbors = None if k <= 0 else torch.full((M, k), -1, dtype=torch.long, device=device)
    n_with_hellinger_match = 0

    # Single loop to build both outputs
    for d in range(M):
        if counts[d] == 0:
            continue

        # Get this detection's candidates
        start, end = start_indices[d].item(), end_indices[d].item()
        h_vals = H2_sorted[start:end]
        o_vals = obj_sorted[start:end]

        # Filter by threshold once
        valid = h_vals < hellinger_thresh
        if not valid.any():
            continue

        h_vals = h_vals[valid]
        o_vals = o_vals[valid]
        order_by_h = torch.argsort(h_vals)
        h_vals = h_vals[order_by_h]
        o_vals = o_vals[order_by_h]

        # Build variable-length neighbor list (all valid neighbors), sorted by
        # Hellinger so downstream best-only correspondence can use neighbor[0].
        neighbors[d] = o_vals
        n_with_hellinger_match += 1

        # Build fixed top-k if requested
        if k > 0:
            kk = min(k, h_vals.shape[0])
            k_neighbors[d, :kk] = o_vals[:kk]

    if return_diagnostics:
        diag = {
            "n_detections": int(M),
            "n_state_objects": int(N),
            "n_active_detections": int(n_active_dets),
            "n_with_class_feat_candidates": int(n_with_class_feat_candidates),
            "n_with_hellinger_match": int(n_with_hellinger_match),
        }
        return neighbors, k_neighbors, diag
    return neighbors, k_neighbors


@torch.no_grad()
def get_neighbors_by_hellinger_distance(
    detections_dict,
    db_dict,
    hellinger_thresh=0.8,
    active_mask=None,
    k=0,
    eps_cov=DEFAULT_HELLINGER_MATCH_FLOOR,
):
    """
    Neighbor search that uses only Hellinger distance.
    """
    det_means = detections_dict["means"]
    det_cov6 = detections_dict["cov6"]
    db_means = db_dict["means"]
    db_cov6 = db_dict["cov6"]
    device = det_means.device

    M = det_means.shape[0]
    N = db_means.shape[0] if db_means is not None else 0

    empty_neighbors = [torch.empty(0, dtype=torch.long, device=device) for _ in range(M)]
    empty_topk = None if k <= 0 else torch.full((M, k), -1, dtype=torch.long, device=device)
    if N == 0 or M == 0:
        return empty_neighbors, empty_topk

    if active_mask is None:
        active_mask = torch.ones(M, dtype=torch.bool, device=device)
    else:
        active_mask = active_mask.to(device=device, dtype=torch.bool)

    db_active = db_dict.get("active")
    if db_active is None:
        db_active = torch.ones(N, dtype=torch.bool, device=device)
    else:
        db_active = db_active.to(device=device, dtype=torch.bool)

    det_cov = _cov6_to_matrix(det_cov6)
    db_cov = _cov6_to_matrix(db_cov6)

    det_finite = torch.isfinite(det_cov).all(dim=(1, 2)) & torch.isfinite(det_means).all(dim=1)
    db_finite = torch.isfinite(db_cov).all(dim=(1, 2)) & torch.isfinite(db_means).all(dim=1)
    det_posdiag = (torch.diagonal(det_cov, dim1=-2, dim2=-1) > 0).all(dim=1)
    db_posdiag = (torch.diagonal(db_cov, dim1=-2, dim2=-1) > 0).all(dim=1)
    det_valid = det_finite & det_posdiag & active_mask
    db_valid = db_finite & db_posdiag & db_active

    det_indices = torch.nonzero(det_valid, as_tuple=False).view(-1)
    db_indices = torch.nonzero(db_valid, as_tuple=False).view(-1)

    if det_indices.numel() == 0 or db_indices.numel() == 0:
        return empty_neighbors, empty_topk

    neighbors = [torch.empty(0, dtype=torch.long, device=device) for _ in range(M)]
    k_neighbors = None if k <= 0 else torch.full((M, k), -1, dtype=torch.long, device=device)

    db_mu = db_means[db_indices]
    db_cov_sel = db_cov[db_indices]

    for d in det_indices.tolist():
        mu_d = det_means[d].unsqueeze(0).expand(db_indices.shape[0], -1)
        cov_d = det_cov[d].unsqueeze(0).expand(db_indices.shape[0], -1, -1)
        with torch.amp.autocast("cuda", enabled=False):
            H2 = _hellinger_distance(mu_d.float(), cov_d.float(), db_mu.float(), db_cov_sel.float(), eps_cov)

        valid = H2 < hellinger_thresh
        if not valid.any():
            continue

        h_vals = H2[valid]
        o_vals = db_indices[valid]
        neighbors[d] = o_vals

        if k > 0:
            kk = min(k, h_vals.shape[0])
            top_k_indices = torch.topk(h_vals, kk, largest=False).indices
            k_neighbors[d, :kk] = o_vals[top_k_indices]

    return neighbors, k_neighbors
