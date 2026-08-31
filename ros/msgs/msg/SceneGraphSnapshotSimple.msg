std_msgs/Header header
uint32 schema_version
uint64 seq

# -------- Objects (parallel arrays; length = num_objects) --------
uint32[] object_ids
float32[] mean_xyz            # length = 3*num_objects, packed [x,y,z,x,y,z,...]
float32[] cov6                # length = 6*num_objects, packed upper-triangular covariance
string[] object_captions

# -------- Viewpoints (ragged, per object) --------
# For object i: viewpoints are in [vp_row_ptr[i], vp_row_ptr[i+1])
# Each viewpoint is SE(3): translation + quaternion (x,y,z,qx,qy,qz,qw)
uint32[] vp_row_ptr           # length = num_objects + 1
float32[] vp_se3              # length = 7 * total_num_viewpoints, packed

# vp_camera_ids are numeric IDs for each viewpoint.
# Convention: index into sorted(mapping.camera_config.CAMERA_CONFIG.keys()).
uint32[] vp_camera_ids        # length = total_num_viewpoints (one cam_id per viewpoint)

# -------- Embeddings (per object) --------
uint16 embedding_dim
float32[] embeddings          # length = num_objects * embedding_dim, row-major

# -------- SigLIP2 CLS embeddings (per object) --------
uint16 siglip2_embedding_dim
float32[] siglip2_embeddings   # length = num_objects * siglip2_embedding_dim, row-major

# -------- Co-visibility graph (ragged adjacency, per object) --------
uint32[] covis_row_ptr        # length = num_objects + 1
uint32[] covis_neighbors      # length = total_num_edges (neighbors as OBJECT IDs)
