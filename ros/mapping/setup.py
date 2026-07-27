"""Ament-python setup for the `mapping` ROS2 package.

This file is used only by colcon / ament_python to build the ROS2 layer. The
pure-Python ``scene_graph`` library is installed separately via the repo-root
``pyproject.toml`` (``uv pip install -e .``).

Standard ament-python layout (Python package name == ROS package name):

    ros/mapping/
    ├── package.xml
    ├── setup.py          (this file)
    ├── setup.cfg
    ├── resource/mapping
    ├── launch/*.launch.py
    └── mapping/          <- Python package
        ├── __init__.py
        ├── nodes/
        └── lib/

``colcon --base-paths ros`` discovers this package and its sibling
``ros/msgs`` (``mapping_msgs``). Do NOT place a ``package.xml`` at ``ros/``
root — colcon would stop recursing there.
"""

from glob import glob

from setuptools import find_packages, setup

package_name = "mapping"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Siming He",
    maintainer_email="siminghe@berkeley.edu",
    description="ROS2 interface for the FARM scene-graph pipeline.",
    license="AGPL-3.0-or-later",
    entry_points={
        "console_scripts": [
            "streaming_mapper = mapping.nodes.streaming_mapper:main",
            "frame_pub = mapping.nodes.frame_pub:main",
            "odin1_depth_pub = mapping.nodes.odin1_depth_pub:main",
            "visual_search_yoloe_text_prompt = mapping.nodes.visual_search_yoloe_text_prompt:main",
        ],
    },
)
