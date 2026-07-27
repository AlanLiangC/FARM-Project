FROM nvidia/cuda:13.0.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=scene_graph
ARG UID=1000
ARG GID=1000

# ── System packages ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo git git-lfs openssh-client ca-certificates \
        curl wget tmux vim nano unzip \
        ffmpeg libsm6 libxext6 libgl1 \
        build-essential cmake ninja-build python3-dev \
        python3-lark python3-empy \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install

# ── uv ───────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# ── ROS 2 Humble ─────────────────────────────────────────────
ARG UBUNTU_CODENAME=jammy
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg2 ca-certificates \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
         ros-humble-ros-base \
         ros-dev-tools \
         python3-ament-package \
         python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Scene-graph-specific ROS2 deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-cv-bridge \
        ros-humble-sensor-msgs \
        ros-humble-image-transport-plugins \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-vision-msgs \
        ros-humble-rviz2 \
        ros-humble-rviz-default-plugins \
    && rm -rf /var/lib/apt/lists/*

# ── Create user ──────────────────────────────────────────────
RUN if ! getent group ${GID} >/dev/null; then groupadd -g ${GID} ${USER}; fi \
    && if ! id -u ${USER} >/dev/null 2>&1; then useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}; fi \
    && usermod -aG sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && chown -R ${UID}:${GID} /home/${USER}

# ── Copy repo ────────────────────────────────────────────────
WORKDIR /home/${USER}/scene_graph
COPY --chown=${UID}:${GID} . .

# ── Python environment via uv ────────────────────────────────
RUN /usr/local/bin/uv venv --python 3.10 --system-site-packages /home/${USER}/.venv

RUN /usr/local/bin/uv pip install \
        --python /home/${USER}/.venv/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision torchaudio \
        "transformers>=4.57.0" \
        -e ./third_party/yoloe \
        -e ./third_party/yoloe/third_party/ml-mobileclip \
        opencv-python h5py huggingface_hub \
        loguru numpy Pillow PyYAML requests \
        "networkx>=2.8" supervision tqdm timm \
        scikit-learn scipy transforms3d \
        "imageio[ffmpeg]" matplotlib pydantic viser \
        "vllm>=0.20.0" accelerate safetensors \
        "qwen-vl-utils>=0.0.14" \
        "setuptools<70" \
        -e . \
    && chown -R ${UID}:${GID} /home/${USER}/.venv

# ── venv → system Python bridge ──────────────────────────────
# `ros2 launch` and every `ros2 run` entry point have shebang `#!/usr/bin/python3`
# (system Python). Heavy deps (torch, scene_graph, numpy, scipy, ...) live only
# in the venv. Drop a .pth that calls site.addsitedir() so system Python picks
# up the venv's site-packages AND recursively processes .pth files inside it
# (needed for editable installs like scene_graph).
#
# Placed in /usr/local/lib/python3.10/dist-packages/ so venv site-packages
# lands BEFORE /usr/lib/python3/dist-packages on sys.path. This is required
# because Debian ships an old numpy 1.21 there that's ABI-incompatible with
# the scipy shipped in the venv; venv's newer numpy must win. Venv setuptools
# is pinned to <70 above to stay compatible with colcon's
# `setup.py develop --editable`.
RUN echo 'import site; site.addsitedir("/home/'"${USER}"'/.venv/lib/python3.10/site-packages")' \
    > /usr/local/lib/python3.10/dist-packages/venv_bridge.pth

# Models are bind-mounted at runtime via docker-compose volumes.
# Run ./bootstrap_models.sh on the host before starting the container.

# ROS2 workspace is built at container startup by entrypoint.sh
# (bind-mounted source overwrites COPY'd files, so build-time colcon is useless)

# ── Shell setup ──────────────────────────────────────────────
RUN echo 'source /home/'"${USER}"'/.venv/bin/activate' >> /home/${USER}/.bashrc \
    && echo 'source /opt/ros/humble/setup.bash' >> /home/${USER}/.bashrc \
    && echo '[ -f /tmp/colcon_ws/install/setup.bash ] && source /tmp/colcon_ws/install/setup.bash' >> /home/${USER}/.bashrc

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TERM=xterm-256color \
    RCUTILS_COLORIZED_OUTPUT=1 \
    SCENE_GRAPH_MODEL_DIR=/home/${USER}/scene_graph/models

USER ${USER}

COPY --chown=${UID}:${GID} docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
