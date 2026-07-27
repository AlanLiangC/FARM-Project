"""Offline drivers for the mapping pipeline.

The offline entry points instantiate the exact same ``StreamingMapper`` node used
online and feed it frames from static data sources (ScanNet ``.sens`` archives,
ROS 2 rosbags, etc.). The algorithm is fully shared with the online path; only
the frame ingestion differs.
"""
