from typing import List, Optional, Set, Tuple

import torch


def _find_parent(parents: torch.Tensor, idx: int) -> int:
    parent = parents[idx].item()
    if parent == idx:
        return idx
    root = _find_parent(parents, parent)
    parents[idx] = root
    return root


def _component_pair_blocked(
    members_a: set[int],
    members_b: set[int],
    cannot_link_pairs: Set[Tuple[int, int]],
) -> bool:
    if not cannot_link_pairs:
        return False
    if len(members_a) > len(members_b):
        members_a, members_b = members_b, members_a
    for a in members_a:
        for b in members_b:
            if (min(a, b), max(a, b)) in cannot_link_pairs:
                return True
    return False


def _union(
    parents: torch.Tensor,
    a: int,
    b: int,
    *,
    cannot_link_pairs: Set[Tuple[int, int]],
    members: Optional[List[set[int]]],
) -> bool:
    root_a = _find_parent(parents, a)
    root_b = _find_parent(parents, b)
    if root_a == root_b:
        return False
    if members is not None and _component_pair_blocked(members[root_a], members[root_b], cannot_link_pairs):
        return False
    if root_a < root_b:
        parents[root_b] = root_a
        if members is not None:
            members[root_a].update(members[root_b])
            members[root_b].clear()
    else:
        parents[root_a] = root_b
        if members is not None:
            members[root_b].update(members[root_a])
            members[root_a].clear()
    return True


def find_object_correspondence(
    neighbors: List[torch.Tensor],
    N: int,
    *,
    cannot_link_pairs: Optional[Set[Tuple[int, int]]] = None,
    assignment_mode: str = "union_all",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return detection winners and per-object canonical winners via union-find."""

    D = len(neighbors)
    det_idx = torch.full((D,), -1)
    parents = torch.arange(N)
    mode = str(assignment_mode or "union_all").strip().lower().replace("-", "_")
    if mode not in {"union_all", "best_only"}:
        raise ValueError(f"Unknown correspondence assignment_mode={assignment_mode!r}")
    blocked_pairs = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in (cannot_link_pairs or set())
        if 0 <= int(a) < N and 0 <= int(b) < N and int(a) != int(b)
    }
    members = [{int(i)} for i in range(N)] if blocked_pairs else None

    # Union objects that share a detection neighbor. ``best_only`` is the
    # standard tracking association ablation: a detection attaches to its
    # nearest valid object but never merges multiple pre-existing objects.
    if mode == "union_all":
        for neighbor in neighbors:
            if neighbor.numel() < 2:
                continue
            base = int(neighbor[0])
            for idx in neighbor[1:]:
                _union(
                    parents,
                    base,
                    int(idx),
                    cannot_link_pairs=blocked_pairs,
                    members=members,
                )

    # Canonical winners per object
    obj_idx = torch.empty(N, dtype=torch.long)
    for i in range(N):
        obj_idx[i] = _find_parent(parents, i)

    # Assign detection winners. Neighbor lists are sorted by Hellinger where
    # produced by get_neighbors; legacy checkpoints still work because
    # union_all keeps the old lowest-index winner convention.
    for det_id, neighbor in enumerate(neighbors):
        if neighbor.numel() == 0:
            continue
        if mode == "best_only":
            det_idx[det_id] = int(neighbor[0])
        else:
            det_idx[det_id] = torch.min(neighbor)

    return det_idx, obj_idx
