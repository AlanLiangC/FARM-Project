from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch

BITS_PER_BLOCK = 64
DEFAULT_COVISIBILITY_MAX_OBJECTS = 1000
DEFAULT_COVISIBILITY_KNN = 10
DEFAULT_COVISIBILITY_SOFT_MEAN_SCALE = 2.0
DEFAULT_COVISIBILITY_WEIGHT_DECAY = 0.9
DEFAULT_COVISIBILITY_EPS_COV = 1e-4


def _blocks_for(max_objects: int) -> int:
    max_objects = max(1, int(max_objects))
    return (max_objects + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK


def _single_bit_tensor(bit_index: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    shift = int(bit_index) % BITS_PER_BLOCK
    one = torch.ones((), dtype=dtype, device=device)
    return torch.bitwise_left_shift(one, shift)


def _state_device(state: Dict[str, Any]) -> torch.device:
    means = state.get("means")
    if isinstance(means, torch.Tensor):
        return means.device
    return torch.device("cpu")


def ensure_covisibility_state(
    state: Dict[str, Any],
    needed_objects: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    device = _state_device(state)
    max_objects = int(state.get("covisibility_max_objects") or DEFAULT_COVISIBILITY_MAX_OBJECTS)
    if needed_objects is not None and int(needed_objects) > max_objects:
        max_objects = int(needed_objects)
    blocks = _blocks_for(max_objects)

    adj = state.get("covisibility_adj_u64")
    active_u64 = state.get("covisibility_active_u64")

    needs_init = not isinstance(adj, torch.Tensor) or adj.ndim != 2
    needs_init = needs_init or not isinstance(active_u64, torch.Tensor) or active_u64.ndim != 1
    needs_init = needs_init or adj.shape[0] < max_objects or adj.shape[1] < blocks

    if needs_init:
        old_adj = adj if isinstance(adj, torch.Tensor) and adj.ndim == 2 else None
        new_adj = torch.zeros((max_objects, blocks), dtype=torch.int64, device=device)
        if old_adj is not None and old_adj.numel() > 0:
            old_adj = old_adj.to(device=device, dtype=torch.int64)
            rows = min(old_adj.shape[0], max_objects)
            cols = min(old_adj.shape[1], blocks)
            new_adj[:rows, :cols] = old_adj[:rows, :cols]
        adj = new_adj
        active_u64 = torch.zeros((blocks,), dtype=torch.int64, device=device)
    else:
        adj = adj.to(device=device, dtype=torch.int64)
        active_u64 = active_u64.to(device=device, dtype=torch.int64)

    state["covisibility_max_objects"] = int(max_objects)
    state["covisibility_blocks"] = int(blocks)
    state["covisibility_adj_u64"] = adj
    state["covisibility_active_u64"] = active_u64
    return adj, active_u64, blocks, max_objects


def _bitmask_from_indices(indices: torch.Tensor, blocks: int) -> torch.Tensor:
    mask = torch.zeros((blocks,), dtype=torch.int64, device=indices.device)
    if indices.numel() == 0:
        return mask
    indices = indices.to(torch.int64)
    if torch.any(indices < 0):
        indices = indices[indices >= 0]
        if indices.numel() == 0:
            return mask
    indices = torch.unique(indices, sorted=False)
    block_idx = torch.div(indices, BITS_PER_BLOCK, rounding_mode="floor")
    bit_idx = torch.remainder(indices, BITS_PER_BLOCK).to(torch.int64)
    bitmask = torch.bitwise_left_shift(torch.ones_like(bit_idx, dtype=torch.int64), bit_idx)
    mask.index_add_(0, block_idx, bitmask)
    return mask


def _ensure_covisibility_weights(state: Dict[str, Any]) -> Dict[int, Dict[int, float]]:
    weights = state.get("covisibility_weights")
    if not isinstance(weights, dict):
        weights = {}
        state["covisibility_weights"] = weights
    return weights


def _sanitize_weight_decay(value: float) -> float:
    try:
        decay = float(value)
    except Exception:
        return DEFAULT_COVISIBILITY_WEIGHT_DECAY
    if decay != decay:  # NaN
        return DEFAULT_COVISIBILITY_WEIGHT_DECAY
    if decay < 0.0:
        return 0.0
    if decay > 1.0:
        return 1.0
    return decay


def _cov6_to_matrix(cov6: torch.Tensor) -> torch.Tensor:
    """
    Convert packed 6D covariance to full 3x3 symmetric matrices.

    cov6: (B, 6) with format [s00, s01, s02, s11, s12, s22]
    """
    if cov6.ndim != 2 or cov6.shape[1] != 6:
        raise ValueError(f"Expected cov6 of shape (B,6), got {tuple(cov6.shape)}")
    b = int(cov6.shape[0])
    cov = torch.zeros((b, 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 1]
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 2]
    cov[:, 1, 1] = cov6[:, 3]
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 4]
    cov[:, 2, 2] = cov6[:, 5]
    return cov


def _hellinger_distance_sq(
    mu1: torch.Tensor,
    cov1: torch.Tensor,
    mu2: torch.Tensor,
    cov2: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """
    Compute Hellinger^2 distance between Gaussian distributions (batched).

    Returns: (K,) Hellinger^2 distances in [0,1] for valid SPD inputs; invalid pairs default to +inf.
    """
    if mu1.ndim != 2 or mu2.ndim != 2 or mu1.shape != mu2.shape or mu1.shape[1] != 3:
        raise ValueError(f"Expected mu1/mu2 of shape (K,3), got {tuple(mu1.shape)} and {tuple(mu2.shape)}")
    if cov1.ndim != 3 or cov2.ndim != 3 or cov1.shape != cov2.shape or cov1.shape[1:] != (3, 3):
        raise ValueError(f"Expected cov1/cov2 of shape (K,3,3), got {tuple(cov1.shape)} and {tuple(cov2.shape)}")

    k = int(cov1.shape[0])
    h2 = torch.full((k,), float("inf"), device=cov1.device, dtype=cov1.dtype)
    if k == 0:
        return h2

    eye = torch.eye(3, device=cov1.device, dtype=cov1.dtype).expand(k, 3, 3)
    cov1 = 0.5 * (cov1 + cov1.transpose(-1, -2)) + float(eps) * eye
    cov2 = 0.5 * (cov2 + cov2.transpose(-1, -2)) + float(eps) * eye
    sigma = 0.5 * (cov1 + cov2)

    if hasattr(torch.linalg, "cholesky_ex"):
        l1, info1 = torch.linalg.cholesky_ex(cov1)
        l2, info2 = torch.linalg.cholesky_ex(cov2)
        l_sigma, info_sigma = torch.linalg.cholesky_ex(sigma)
        valid = (info1 == 0) & (info2 == 0) & (info_sigma == 0)
    else:
        try:
            l1 = torch.linalg.cholesky(cov1)
            l2 = torch.linalg.cholesky(cov2)
            l_sigma = torch.linalg.cholesky(sigma)
            valid = torch.ones((k,), dtype=torch.bool, device=cov1.device)
        except Exception:
            return h2

    if not bool(valid.any()):
        return h2

    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    l1_v = l1[idx]
    l2_v = l2[idx]
    l_sigma_v = l_sigma[idx]
    mu1_v = mu1[idx]
    mu2_v = mu2[idx]

    logdet_cov1 = 2 * torch.log(torch.diagonal(l1_v, dim1=-2, dim2=-1)).sum(-1)
    logdet_cov2 = 2 * torch.log(torch.diagonal(l2_v, dim1=-2, dim2=-1)).sum(-1)
    logdet_sigma = 2 * torch.log(torch.diagonal(l_sigma_v, dim1=-2, dim2=-1)).sum(-1)

    delta = (mu1_v - mu2_v).unsqueeze(-1)
    y = torch.cholesky_solve(delta, l_sigma_v)
    quad = (delta.squeeze(-1) * y.squeeze(-1)).sum(-1)

    log_bc = 0.25 * (logdet_cov1 + logdet_cov2) - 0.5 * logdet_sigma - 0.125 * quad
    bc = torch.exp(log_bc)
    h2[idx] = 1.0 - bc
    return h2


def update_covisibility_active_bitset(state: Dict[str, Any], num_objects: Optional[int] = None) -> torch.Tensor:
    adj, active_u64, blocks, _ = ensure_covisibility_state(state, needed_objects=num_objects)
    active = state.get("active")
    if not isinstance(active, torch.Tensor):
        active_u64.zero_()
        state["covisibility_active_u64"] = active_u64
        return active_u64
    if num_objects is None:
        num_objects = int(active.numel())
    num_objects = max(0, int(num_objects))
    active_slice = active[:num_objects].to(dtype=torch.bool, device=active_u64.device)
    active_u64.zero_()
    if not torch.any(active_slice):
        state["covisibility_active_u64"] = active_u64
        return active_u64
    active_idx = torch.nonzero(active_slice, as_tuple=False).view(-1).to(torch.int64)
    state["covisibility_active_u64"] = _bitmask_from_indices(active_idx, blocks).to(dtype=active_u64.dtype)
    return state["covisibility_active_u64"]


def merge_covisibility_loser_into_winner(
    state: Dict[str, Any],
    loser_idx: int,
    winner_idx: int,
    *,
    num_objects: int,
) -> None:
    if loser_idx == winner_idx:
        return
    if loser_idx < 0 or winner_idx < 0:
        return
    num_objects = max(0, int(num_objects))
    adj, _active_u64, blocks, _ = ensure_covisibility_state(state, needed_objects=num_objects)
    if loser_idx >= num_objects or winner_idx >= num_objects:
        return

    adj_slice = adj[:num_objects, :blocks]
    adj_slice[winner_idx] |= adj_slice[loser_idx]

    loser_block = loser_idx // BITS_PER_BLOCK
    winner_block = winner_idx // BITS_PER_BLOCK
    loser_bit_t = _single_bit_tensor(loser_idx % BITS_PER_BLOCK, dtype=adj_slice.dtype, device=adj_slice.device)
    winner_bit_t = _single_bit_tensor(winner_idx % BITS_PER_BLOCK, dtype=adj_slice.dtype, device=adj_slice.device)

    rows_with_loser = (adj_slice[:, loser_block] & loser_bit_t) != 0
    if torch.any(rows_with_loser):
        rows_idx = torch.nonzero(rows_with_loser, as_tuple=False).view(-1)
        if rows_idx.numel() > 0:
            adj_slice[rows_idx, winner_block] |= winner_bit_t

    adj_slice[:, loser_block] &= ~loser_bit_t
    adj_slice[loser_idx].zero_()
    adj_slice[winner_idx, winner_block] &= ~winner_bit_t

    weights = state.get("covisibility_weights")
    if not isinstance(weights, dict):
        return
    loser_weights = weights.get(loser_idx)
    if not isinstance(loser_weights, dict):
        return
    winner_weights = weights.get(winner_idx)
    if not isinstance(winner_weights, dict):
        winner_weights = {}
        weights[winner_idx] = winner_weights

    for neighbor_idx, weight in list(loser_weights.items()):
        try:
            neighbor_idx_int = int(neighbor_idx)
        except Exception:
            continue
        if neighbor_idx_int in (winner_idx, loser_idx):
            continue
        try:
            weight_val = float(weight)
        except Exception:
            continue
        winner_weights[neighbor_idx_int] = float(winner_weights.get(neighbor_idx_int, 0.0)) + weight_val

        neighbor_weights = weights.get(neighbor_idx_int)
        if isinstance(neighbor_weights, dict):
            neighbor_weights.pop(loser_idx, None)
            neighbor_weights[winner_idx] = float(neighbor_weights.get(winner_idx, 0.0)) + weight_val

    weights.pop(loser_idx, None)


@torch.no_grad()
def update_covisibility_from_visible_indices(
    state: Dict[str, Any],
    visible_indices: Iterable[int],
    *,
    num_objects: Optional[int] = None,
    k: int = DEFAULT_COVISIBILITY_KNN,
    soft_mean_scale: float = DEFAULT_COVISIBILITY_SOFT_MEAN_SCALE,
    weight_decay: float = DEFAULT_COVISIBILITY_WEIGHT_DECAY,
    eps_cov: float = DEFAULT_COVISIBILITY_EPS_COV,
) -> None:
    """
    Update the covisibility adjacency using the definition:

    Two objects are covisible iff both appear in the same *batch* of frames (i.e., provided together in
    `visible_indices`) and they are connected via k-nearest-neighbors (by Gaussian spatial Hellinger^2 distance)
    among the objects visible in that batch, with the symmetric soft cutoff:

        keep edge (i,j) iff d(i,j) <= soft_mean_scale * max(median(d_i), median(d_j))

    where d(i,j) is the Gaussian Hellinger^2 distance and d_i are i's kNN distances.
    """
    visible_list = sorted({int(i) for i in visible_indices if i is not None})
    if len(visible_list) < 2:
        return
    max_idx = max(visible_list)
    needed = max_idx + 1
    if num_objects is not None:
        needed = max(needed, int(num_objects))
    adj, _active_u64, blocks, _ = ensure_covisibility_state(state, needed_objects=needed)
    device = adj.device
    if num_objects is None:
        num_objects = int(state.get("means").shape[0]) if isinstance(state.get("means"), torch.Tensor) else needed
    num_objects = max(0, int(num_objects))

    valid_list = [i for i in visible_list if 0 <= i < num_objects]
    if len(valid_list) < 2:
        return

    means = state.get("means")
    cov6 = state.get("cov6")
    if not isinstance(means, torch.Tensor) or not isinstance(cov6, torch.Tensor):
        return
    if means.ndim != 2 or means.shape[1] != 3 or cov6.ndim != 2 or cov6.shape[1] != 6:
        return
    if means.shape[0] < num_objects or cov6.shape[0] < num_objects:
        return

    idx_tensor = torch.as_tensor(valid_list, dtype=torch.int64, device=device)
    idx_tensor = torch.unique(idx_tensor, sorted=True)
    if idx_tensor.numel() < 2:
        return
    adj_slice = adj[:num_objects, :blocks]
    weights = _ensure_covisibility_weights(state)
    decay = _sanitize_weight_decay(weight_decay)

    try:
        means_sel = means.index_select(0, idx_tensor).to(device=device)
        cov6_sel = cov6.index_select(0, idx_tensor).to(device=device)
    except Exception:
        return

    with torch.amp.autocast("cuda", enabled=False):
        means_sel_f = means_sel.float()
        cov_sel = _cov6_to_matrix(cov6_sel.float())
        m = int(idx_tensor.numel())
        k_eff = min(max(0, int(k)), max(0, m - 1))
        if k_eff <= 0:
            return
        ii, jj = torch.triu_indices(m, m, offset=1, device=device)
        if ii.numel() == 0:
            return
        h2 = _hellinger_distance_sq(
            means_sel_f.index_select(0, ii),
            cov_sel.index_select(0, ii),
            means_sel_f.index_select(0, jj),
            cov_sel.index_select(0, jj),
            eps=float(eps_cov),
        )

    dist = torch.full((m, m), float("inf"), device=device, dtype=h2.dtype)
    dist[ii, jj] = h2
    dist[jj, ii] = h2
    dist.fill_diagonal_(float("inf"))

    # Directed kNN then symmetrized for undirected covisibility.
    vals, nn = torch.topk(dist, k=k_eff, dim=1, largest=False)
    finite = torch.isfinite(vals)
    if not bool(finite.any()):
        return

    scale = float(soft_mean_scale)
    if scale != scale or scale <= 0:
        scale = float("inf")
    if scale != float("inf"):
        vals_f = torch.where(finite, vals, float("inf"))
        vals_sorted, _ = torch.sort(vals_f, dim=1)
        counts = finite.sum(dim=1)
        med = torch.full((m,), float("inf"), device=device, dtype=vals.dtype)
        has_any = counts > 0
        if bool(has_any.any()):
            mid = torch.div((counts - 1).clamp(min=0), 2, rounding_mode="floor")
            rows_med = torch.arange(m, device=device)[has_any]
            med[has_any] = vals_sorted[rows_med, mid[has_any]]

        med_i = med.unsqueeze(1).expand_as(nn)
        med_j = med.index_select(0, nn.reshape(-1)).view_as(nn)
        thresh = scale * torch.maximum(med_i, med_j)
        keep = finite & (vals <= thresh)
        if not bool(keep.any()):
            return
    else:
        keep = finite

    rows = torch.arange(m, device=device).unsqueeze(1).expand_as(nn)
    adj_local = torch.zeros((m, m), dtype=torch.bool, device=device)
    adj_local[rows[keep], nn[keep]] = True
    # Avoid in-place ops with overlapping memory (transpose is a view).
    adj_local = adj_local | adj_local.transpose(0, 1)

    for local_i in range(m):
        neighbors_local = torch.nonzero(adj_local[local_i], as_tuple=False).view(-1)
        if neighbors_local.numel() == 0:
            continue
        neighbors_global = idx_tensor.index_select(0, neighbors_local)
        row_mask = _bitmask_from_indices(neighbors_global, blocks).to(dtype=adj_slice.dtype)
        obj_idx = int(idx_tensor[local_i].item())
        adj_slice[obj_idx] |= row_mask
        block = obj_idx // BITS_PER_BLOCK
        bit_t = _single_bit_tensor(obj_idx % BITS_PER_BLOCK, dtype=adj_slice.dtype, device=adj_slice.device)
        adj_slice[obj_idx, block] &= ~bit_t

        weights_row = weights.setdefault(obj_idx, {})
        for neighbor_idx_t in neighbors_global.tolist():
            neighbor_idx = int(neighbor_idx_t)
            if neighbor_idx == obj_idx or neighbor_idx < obj_idx:
                continue
            prev = float(weights_row.get(neighbor_idx, 0.0))
            updated = decay * prev + 1.0
            weights_row[neighbor_idx] = updated
            neighbor_row = weights.setdefault(neighbor_idx, {})
            neighbor_row[obj_idx] = updated
