# Match between a visual search goal and an observed local caption.

# Goal object id (-1 if unspecified).
int64 goal_object_id

# Natural-language goal description / query.
string goal_description

# Matched local caption (detection) from the current frame.
LocalCaption local_caption

## Convenience fields copied from `local_caption`.

# 2D bounding box in pixel coordinates.
vision_msgs/BoundingBox2D bbox

# 3D position of object in VisualSearchResultArray.header.frame_id (typically map/world).
geometry_msgs/Point position_3d
