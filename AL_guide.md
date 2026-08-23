# FARM 本机 Conda 运行指南（无 Docker）

本文记录本机已验证的 FARM 原生运行方式。上游 README 只支持 Docker；这里的
Conda + 系统 ROS 2 组合是针对当前实验机的适配方案。

## 1. 已验证配置

验证日期：2026-08-20。

| 项目 | 本机配置 |
| --- | --- |
| 项目目录 | `/home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project` |
| 数据目录 | `/home/alan/AlanLiang/Projects/pure_checkpoints/FARM-Scenes` |
| Conda 环境 | `farm`，Python 3.10.20 |
| GPU | NVIDIA GeForce RTX 4070 Laptop，8 GB |
| 驱动 | 580.173.02（最高支持 CUDA 13.0） |
| PyTorch | 2.9.1+cu128，torchvision 0.24.1+cu128，torchaudio 2.9.1+cu128 |
| ROS | ROS 2 Humble，`/opt/ros/humble` |
| Transformers | 4.57.6 |

当前已完成的闭环验证：

- PyTorch CUDA 张量测试通过；
- `mapping_msgs` 和 `mapping` 已由本机 ROS 2 构建；
- FARM-Scenes warehouse 的 6 个真实帧已完成 YOLOE + DINOv3 + 3D 融合 +
  co-visibility 构图；
- 输出图可以重新加载：6 个 active objects、特征形状 `(6, 384)`、6 个图像记录、
  4,087 个体素键，3D 坐标均为有限值；
- 验证产物位于
  `/home/alan/AlanLiang/Projects/pure_checkpoints/FARM-Scenes/out/al_warehouse_smoke.pt`。

受 8 GB 显存限制，本轮没有安装或启动 vLLM，也没有运行 caption、文本查询和区域
LLM 标注。离线几何 scene graph 构造不依赖这些服务。

## 2. 每次打开终端时初始化

后文所有命令都默认先执行这一段：

```bash
source /home/alan/miniconda3/etc/profile.d/conda.sh
conda activate farm
source /opt/ros/humble/setup.bash

export FARM_ROOT=/home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project
export FARM_DATA=/home/alan/AlanLiang/Projects/pure_checkpoints/FARM-Scenes
export FARM_ROS_VENDOR="$CONDA_PREFIX/opt/ros/humble"

# 本机没有 sudo 权限，vision_msgs 被解包到了 Conda 环境；这两行使其可见。
export PYTHONPATH="$FARM_ROS_VENDOR/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FARM_ROS_VENDOR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$FARM_ROOT"
source install/setup.bash
export SCENE_GRAPH_MODEL_DIR="$FARM_ROOT/models"
```

快速检查：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -c 'from mapping_msgs.msg import RGBDFrame, LocalCaption; from mapping.nodes.streaming_mapper import StreamingMapper; print("ROS/FARM imports OK")'
```

预期第一条显示 `2.9.1+cu128 12.8 True`，第二条显示
`ROS/FARM imports OK`。

## 3. 已创建环境的安装记录

环境已经建好，正常运行不需要重复本节。需要从零重建时使用以下命令。

### 3.1 Conda 和 Python 依赖

```bash
conda create -y -n farm python=3.10 pip
conda activate farm

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128

cd /home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project
python -m pip install \
  'transformers==4.57.6' \
  -e ./third_party/yoloe \
  -e ./third_party/yoloe/third_party/ml-mobileclip \
  -e '.[dev]' \
  pydantic 'setuptools<70'

# ROS 2 Humble 的接口生成器要求 empy 3.x，不能使用 empy 4.x。
python -m pip install 'empy==3.3.4' lark catkin_pkg
```

不要在这个 8 GB 环境里执行 `pip install '.[vllm]'`。vLLM 及其多个模型服务会显著
增加磁盘和显存占用，而且本轮离线构图不需要它。

### 3.2 模型

仓库自带 `models/dinov3-vits16`。下面的命令下载离线构图需要的两个 YOLOE 权重和
MobileCLIP 权重，同时跳过本轮不需要的 3.4 GB SigLIP2：

```bash
cd /home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project
./bootstrap_models.sh --skip-siglip2
```

验证文件：

```bash
ls -lh \
  models/yoloe/yoloe-v8l-seg.pt \
  models/yoloe/yoloe-v8l-seg-pf.pt \
  models/mobileclip/mobileclip_blt.pt \
  models/dinov3-vits16/model.safetensors
```

当前使用仓库自带的非 gated DINOv3 ViT-S/16。运行时关于缺少
`dinov3-vits16plus` 的提示是性能等级警告，不影响构图；只有复现论文级合并效果时
才需要申请 gated 模型。

### 3.3 ROS 2 依赖和工作区构建

系统已有 ROS 2 Humble 和 `cv_bridge`，但最初缺少 `vision_msgs`。有 sudo 权限时
推荐直接安装：

```bash
sudo apt update
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge \
  ros-humble-image-transport-plugins ros-humble-rosbag2-storage-mcap
```

本机当前没有 sudo 权限，因此实际采用了只写 Conda 环境的安装方式：

```bash
farm_deb_dir=$(mktemp -d)
cd "$farm_deb_dir"
apt download ros-humble-vision-msgs
dpkg-deb -x ros-humble-vision-msgs_*.deb "$CONDA_PREFIX"
cd /home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project
```

构建 FARM 的 ROS 消息和 Python 节点：

```bash
source /opt/ros/humble/setup.bash
export FARM_ROS_VENDOR="$CONDA_PREFIX/opt/ros/humble"
export CMAKE_PREFIX_PATH="$FARM_ROS_VENDOR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export PYTHONPATH="$FARM_ROS_VENDOR/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$FARM_ROS_VENDOR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd /home/alan/AlanLiang/Projects/AlanLiang/FARM-Navigation/FARM-Project
colcon build --base-paths ros/msgs ros/mapping --symlink-install
source install/setup.bash
```

Conda 和 Ubuntu 的 `libpython3.10` 路径可能触发 CMake runtime search path warning；
本机最终显示 `2 packages finished`，并且 Python 导入测试通过，因此该 warning 可以
忽略。

## 4. 离线 scene graph 构造

### 4.1 已跑通的最小真实数据闭环

先执行第 2 节的终端初始化，然后运行：

```bash
mkdir -p "$FARM_DATA/out"

python -m scene_graph.offline.run \
  --source frames-json \
  --frames-json-dir "$FARM_DATA/scenes/grandtour/2024-11-25_warehouse" \
  --frames-json-camera hdr_front \
  --end 6 \
  --target-fps 0 \
  --save-path "$FARM_DATA/out/al_warehouse_smoke.pt" \
  --covisibility \
  --no-mask-observations \
  --log-every 1 \
  --extra-param segmenter_imgsz:=512 \
  --extra-param segmenter_dino_load_size:=384 \
  --extra-param scene_graph_json_save_enabled:=false \
  --extra-param scene_graph_snapshot_save_enabled:=false
```

说明：

- 没有 `--caption`，因此不会连接 vLLM；
- 没有 `--viser`，避免实时网页可视化降低约 6 倍吞吐；
- 512/384 是针对本机 8 GB 显存的保守尺寸；
- `scene_state_device` 默认是 CPU，历史对象不会持续占用显存；
- 日志中的 `SigLIP2 text embedding service disabled` 在关闭 caption 时是正常现象；
- `--end` 对每个选中的相机分别计数。本命令只选 `hdr_front`，所以总计恰好 6 帧。

本机实测首帧包含模型初始化，约 1.5 秒；其余 5 帧平均约 19 FPS。成功结束时日志应
包含：

```text
Done. 5 frames in 0.3s (19.01 fps avg)
Saved scene state to .../out/al_warehouse_smoke.pt
```

这里的 `5 frames` 不包含单独统计的 1 个 warm-up frame，总处理数仍为 6。

### 4.2 校验构图产物

`scene_state.pt` 使用 PyTorch pickle；只加载自己生成或可信来源的文件。

```bash
python - <<'PY'
from scene_graph.scene_state_io import load_scene_state

path = "/home/alan/AlanLiang/Projects/pure_checkpoints/FARM-Scenes/out/al_warehouse_smoke.pt"
state = load_scene_state(path, feature_dim=384, device="cpu")
active = state["active"].bool()
summary = {
    "objects_total": int(state["means"].shape[0]),
    "objects_active": int(active.sum()),
    "feature_shape": tuple(state["features"].shape),
    "images": len(state["images"]),
    "counts": state["count"][active].tolist(),
    "voxel_keys": int(state["object_voxel_keys_flat"].numel()),
    "covisibility_nodes": len(state["covisibility_weights"]),
    "finite_means": bool(state["means"][active].isfinite().all()),
}
print(summary)
PY
```

本机当前输出：

```text
{'objects_total': 6, 'objects_active': 6, 'feature_shape': (6, 384),
 'images': 6, 'counts': [6, 5, 6, 4, 2, 2], 'voxel_keys': 4087,
 'covisibility_nodes': 6, 'finite_means': True}
```

`count > 1` 证明对象确实在多个帧间进行了关联和融合，而不只是逐帧检测后写空文件。

### 4.3 扩大运行规模

先以单相机 100 帧继续验证温度、显存和输出质量：

```bash
python -m scene_graph.offline.run \
  --source frames-json \
  --frames-json-dir "$FARM_DATA/scenes/grandtour/2024-11-25_warehouse" \
  --frames-json-camera hdr_front \
  --end 100 \
  --target-fps 0 \
  --save-path "$FARM_DATA/out/warehouse_front_100.pt" \
  --covisibility --no-mask-observations \
  --extra-param segmenter_imgsz:=512 \
  --extra-param segmenter_dino_load_size:=384
```

确认稳定后，去掉 `--frames-json-camera` 和 `--end` 即可处理 warehouse 的全部 2,553
帧。长任务建议保留 `--no-mask-observations`；如果后续需要逐对象的 2D mask/crop
证据，再去掉该参数。运行期间可用以下命令监控：

```bash
watch -n 1 nvidia-smi
```

如果仍然 OOM，按顺序尝试：关闭其他 GPU 图形程序、把 `segmenter_imgsz` 降至 448、
把 `segmenter_dino_load_size` 降至 320。不要先关闭 DINO；关闭它会改变跨帧对象合并
特征和最终图的质量。

## 5. 查看已有或新生成的 scene graph

查看刚生成的图：

```bash
python scripts/view_scene_state.py \
  --pt "$FARM_DATA/out/al_warehouse_smoke.pt" \
  --cloud "$FARM_DATA/scenes/grandtour/2024-11-25_warehouse/cloud.npz"
```

浏览器打开 <http://localhost:8080>。也可以直接查看数据集提供的完整预构建图：

```bash
python scripts/view_scene_state.py \
  --pt "$FARM_DATA/scene_graphs/grandtour/2024-11-25_warehouse.pt" \
  --cloud "$FARM_DATA/scenes/grandtour/2024-11-25_warehouse/cloud.npz"
```

预构建图较大，首次向浏览器传输点云可能需要几十秒。查看 3D 图不需要 vLLM；Query
面板中的自然语言检索需要 LLM 和 embedding 服务，本机当前未启用。

## 6. ROS 2 在线运行边界

Conda 环境和 ROS 消息包已经能导入，但本轮没有连接真实相机或机器人，因此只验证了
共用 `StreamingMapper` 的离线入口，没有宣称在线传感器链路已验证。

需要接 Spot 风格的 5 相机话题时，先执行第 2 节初始化，再关闭 caption 启动：

```bash
ros2 launch mapping mapping_five_cam.launch.py caption_enabled:=false \
  segmenter_device:=cuda:0
```

相机话题映射在 `src/scene_graph/camera_config.py`。如果实际设备话题与默认 Spot 配置
不同，先修改映射，再启动。回放 bag 时另开终端并执行相同的第 2 节初始化，然后：

```bash
ros2 bag play /path/to/bag --clock --rate 0.5
```

只有 bag 已经包含 `/mapping/rgbd_frame/<camera>` 类型的 `RGBDFrame` 时，才使用
`python -m scene_graph.offline.run --source rosbag`。普通 RGB/depth/TF bag 应使用在线
launch + `ros2 bag play`。

## 7. vLLM、caption 和文本查询（本机暂缓）

FARM 默认完整服务包括 caption/query LLM、文本 embedding 和 VL embedding，官方建议
约 50 GB 总显存。当前 8 GB GPU 不适合同时启动这些服务；即使
`pure_checkpoints/Qwen3-VL-4B-Instruct` 已下载，它也不能替代 FARM 默认需要的全部
三个服务。

后续有多卡服务器时再做以下工作：

1. 在独立环境或服务器上安装 `.[vllm]`；
2. 准备 README 所列的 Qwen3.5-9B、Qwen3-Embedding-0.6B 和
   Qwen3-VL-Embedding-2B；
3. 通过 `VLLM_BASE_URL`、`VLLM_EMBED_BASE_URL`、
   `VLLM_QWEN3_VL_EMBED_BASE_URL` 指向远程服务；
4. 构图时加入 `--caption`，或使用 `scripts/query_scene_graph.py` 查询已有图。

在这些服务可用之前，不要给当前离线命令添加 `--caption`，否则退出前会等待 caption
队列并最终超时。

## 8. 常见问题

### `ModuleNotFoundError: No module named 'rclpy'`

必须先 `conda activate farm`，再 `source /opt/ros/humble/setup.bash`。

### 找不到 `mapping_msgs` 或 `mapping`

在项目根目录执行 `source install/setup.bash`。若 `install/` 不存在，按 3.3 节重新
`colcon build`。

### 找不到 `vision_msgs` 或加载其 `.so` 失败

确认第 2 节的 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 已设置。若 Conda 环境被重建，需重新
执行 3.3 节的 apt 安装或无 sudo 解包步骤。

### `ModuleNotFoundError: No module named 'em'`

```bash
python -m pip install 'empy==3.3.4'
```

### DINOv3 从 Hugging Face 联网或提示找不到权重

确认已在项目根目录设置：

```bash
export SCENE_GRAPH_MODEL_DIR="$PWD/models"
test -f "$SCENE_GRAPH_MODEL_DIR/dinov3-vits16/model.safetensors"
```

### `CUDA out of memory`

关闭 vLLM/其他 GPU 进程，保持 `--no-mask-observations`，使用本文的 512/384 设置；仍
不足时按 4.3 节逐级降到 448/320。用 `nvidia-smi` 确认不是桌面以外的程序占用显存。
