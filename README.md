# Frustum3D-Lite

轻量级、端到端的 360° 多传感器 3D 目标检测框架。

Light3D（Frustum3D-Lite）使用多相机的 2D 检测框引导 LiDAR 点云裁剪（frustum），通过轻量化的 PointNet 风格回归网络直接预测 3D 包围盒（cx, cy, cz, w, l, h, yaw），无需 BEV 投影或深度估计，适合资源受限的移动平台与竞赛场景。

> 项目状态：进行中 — 目标：实时 360° 检测、轻量化部署与多数据集支持（nuScenes / KITTI）。

## 特性

- 360° 全向检测：支持多相机（默认 6 个）+ LiDAR，多传感器融合覆盖车辆周边场景。
- 端到端流水线：YOLO 2D 检测 → frustum 点云裁剪 → 去噪/聚类 → PointNet 回归 3D BBox。
- 轻量网络：模型参数量约 50K，适合单卡或嵌入式平台推理。
- 可配置：通过 YAML 切换数据集（nuScenes / KITTI）与传感器标定（相机‑LiDAR 内外参）。

## 效果示例

左：360° LiDAR 俯视图（红=自身点）；右：6 相机 YOLO 2D 检测与回归出的 3D BBox。

![demo](docs/images/demo.gif)

## 快速开始

1. 安装依赖（建议在虚拟环境或容器内运行）

```bash
pip install -r requirements.txt
# 如果使用 ONNX/pyrealsense 等工具，请根据需要安装 tools/requirements.txt
pip install -r tools/requirements.txt
```

2. 预处理（示例：5 sweep）

```bash
python scripts/tools/preprocess_phase3.py --nsweeps 5
```

3. 训练（示例配置）

```bash
python scripts/train/train_phase3.py --config config/train.yaml --epochs 80
```

4. 可视化/推理

```bash
# 360° 可视化
python scripts/test/visualize_360.py

# 单场景推理（参考 pipeline/inference.py）
python pipeline/inference.py --config config/train.yaml --scene /path/to/scene
```

## 目录结构（概要）

```
config/                # 训练与传感器标定配置（train.yaml, sensor.yaml, phase3.yaml）
pipeline/              # 推理/融合管线：detector, projector, fusion, pointnet, preprocess
scripts/               # 训练、测试与预处理脚本
src/                   # C++ 占位（CUDA/加速实现）
display/               # 可视化输出（通常被 gitignore）
```

**流程概览**：多相机图像 → YOLO 2D 检测 → frustum 裁剪 LiDAR → 地面去除 + 去噪 + DBSCAN 聚类 → 采样点 (e.g. 512) → PointNet3DDetector 回归 3D BBox。

## 配置要点

- `config/train.yaml`：训练参数、数据路径、超参
- `config/sensor.yaml`：相机/LiDAR 内外参与检测范围
- 支持通过 YAML 切换数据集（nuScenes / KITTI）

示例：
```yaml
dataset:
  name: nuscenes   # nuscenes | kitti
  root: data/nuscenes
  version: v1.0-mini
```

## 模型概览

- 骨干：PointNet 风格的点云特征提取
- 输入特征：backbone_feat + prior/centroid/extent/viewdir/face_cov/bbox_feat 等。
  - 注意：`face_cov`（16 dim）在推理阶段会被置为 0，用作占位/未来扩展，推理时并不参与有效计算。
  - 合并后特征维度示例：220（其中 face_cov 16 dim 在推理时为 0，实际有效维度约 204）。
- 每个任务分头回归中心、尺寸与朝向

## 坐标与约定

- LiDAR 坐标系：X = 右, Y = 前, Z = 上
- 尺寸格式：`[width, length, height]`（nuScenes 约定）

## 贡献与计划

欢迎提交 Issue 与 PR：
- 计划项：多目标跟踪、轨迹预测、占用网格、更多数据集适配与推理加速。

如需我把 README 翻译成英文版或增加安装/单测/CI 示例，我可以继续修改.
