import numpy as np

from scene_graph.retrieval.spatial_reasoning.predicates import PredicateEvaluator


def _evaluator(scene_state):
    return PredicateEvaluator(scene_state, llm=object(), embedder=object(), verbose=False)


def test_hm3d_scene_state_uses_y_as_vertical_axis():
    # Objects vertically stacked along +y (the HM3D up-axis), horizontally
    # aligned. If the evaluator wrongly kept z as vertical, the pair would
    # read as neither aligned nor stacked and both assertions would fail.
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "images": [{"source_ref": "/data/iref_vla/rendered_trajectory_magnet/scene/frames_000.npz#frame=0"}],
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.vertical_axis == 1
    assert evaluator.evaluate("Below", 0, 1, use_vlm=False).score > 0.99
    assert evaluator.evaluate("Above", 1, 0, use_vlm=False).score > 0.99


def test_default_scene_state_uses_z_as_vertical_axis():
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.vertical_axis == 2
    assert evaluator.evaluate("Below", 0, 1, use_vlm=False).score > 0.99
    assert evaluator.evaluate("Above", 1, 0, use_vlm=False).score > 0.99


def test_above_below_require_horizontal_alignment():
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.evaluate("Above", 1, 0, use_vlm=False).score == 0.0
    assert evaluator.evaluate("Below", 0, 1, use_vlm=False).score == 0.0


def test_left_right_use_shared_mask_image_plane():
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "object_mask_observations": [
            [
                {
                    "image_id": 7,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [20, 30, 60, 80],
                    "raw_pixels": 1200,
                    "score": 0.9,
                }
            ],
            [
                {
                    "image_id": 7,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [120, 30, 160, 80],
                    "raw_pixels": 1200,
                    "score": 0.9,
                }
            ],
        ],
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.evaluate("LeftOf", 0, 1, use_vlm=False).score > 0.99
    assert evaluator.evaluate("LeftOf", 1, 0, use_vlm=False).score < 0.01
    assert evaluator.evaluate("RightOf", 1, 0, use_vlm=False).score > 0.99


def test_left_right_without_shared_mask_view_scores_zero_when_masks_exist():
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, -2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "object_mask_observations": [
            [
                {
                    "image_id": 7,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [20, 30, 60, 80],
                    "raw_pixels": 1200,
                }
            ],
            [
                {
                    "image_id": 8,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [120, 30, 160, 80],
                    "raw_pixels": 1200,
                }
            ],
            [],
        ],
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.evaluate("LeftOf", 0, 1, use_vlm=False).score == 0.0
    assert evaluator.evaluate("RightOf", 1, 0, use_vlm=False).score == 0.0


def test_front_behind_use_shared_frame_camera_depth():
    scene_state = {
        "means": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        "images": [
            {
                "image_id": 0,
                "pose": np.eye(4, dtype=np.float32),
            }
        ],
        "object_mask_observations": [
            [
                {
                    "image_id": 0,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [20, 30, 60, 80],
                    "raw_pixels": 1200,
                }
            ],
            [
                {
                    "image_id": 0,
                    "image_shape": [100, 200],
                    "raw_bbox_xyxy": [120, 30, 160, 80],
                    "raw_pixels": 1200,
                }
            ],
        ],
    }

    evaluator = _evaluator(scene_state)

    assert evaluator.evaluate("InFrontOf", 0, 1, use_vlm=False).score > 0.95
    assert evaluator.evaluate("Behind", 1, 0, use_vlm=False).score > 0.95
    assert evaluator.evaluate("InFrontOf", 1, 0, use_vlm=False).score < 0.05
