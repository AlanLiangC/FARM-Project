# Local captions for one image.

std_msgs/Header header
# header.frame_id should be the world/map frame for 3D positions.

# Robot pose at observation time, expressed in header.frame_id.
geometry_msgs/Pose robot_pose

# Original image used for detection (camera optical frame).
sensor_msgs/Image image

# Identifier for the camera stream that produced this frame.
string camera_name

# List of local captions for this frame.
LocalCaption[] local_captions
