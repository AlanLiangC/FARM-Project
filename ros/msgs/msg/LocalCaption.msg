# Single detection with a local caption.

# 2D bounding box in pixel coordinates.
# BoundingBox2D uses center (x, y, theta) and size_x/size_y in pixels.
vision_msgs/BoundingBox2D bbox

# Detection confidence (0-1).
float32 confidence

# Semantic category from YOLOE.
string category

# 3D position of object in LocalCaptionArray.header.frame_id (typically map/world).
geometry_msgs/Point position_3d

# Caption from captioning (empty if unmatched).
string caption

# Caption embedding (e.g., Qwen3-embedding vector for `caption`; empty if unavailable).
float32[] caption_embedding

# Persistent object id if this detection is matched to a past object, else -1.
int64 matched_object_id

# If `matched_object_id` has absorbed other objects via scene-graph merges, list those
# merged-away ("loser") ids here so downstream consumers can match stale ids too.
int64[] matched_object_loser_ids
