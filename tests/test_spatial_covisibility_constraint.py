from scene_graph.retrieval.spatial_reasoning.executor import _build_covisibility_constraint
from scene_graph.retrieval.spatial_reasoning.methods import get_spatial_method


def test_covisibility_constraint_uses_k_hop_shared_image_graph() -> None:
    state = {
        "means": [[0.0, 0.0, 0.0]] * 4,
        "object_image_ids": [
            [10],
            [10, 11],
            [11, 12],
            [12],
        ],
    }

    one_hop = _build_covisibility_constraint(state, hops=1, verbose=False)
    assert one_hop is not None
    assert one_hop.filter_anchors(0, [1, 2, 3]) == [1]

    two_hop = _build_covisibility_constraint(state, hops=2, verbose=False)
    assert two_hop is not None
    assert two_hop.filter_anchors(0, [1, 2, 3]) == [1, 2]

    three_hop = _build_covisibility_constraint(state, hops=3, verbose=False)
    assert three_hop is not None
    assert three_hop.filter_anchors(0, [1, 2, 3]) == [1, 2, 3]


def test_covisibility_method_profile_registered() -> None:
    method = get_spatial_method("covis3")
    assert method.name == "unified_soft_w50_covis3"
    assert method.covisibility_hops == 3
    assert method.covisibility_source == "shared_images"
    assert method.class_mismatch_floor == 0.3
