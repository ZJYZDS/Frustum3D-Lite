# Frustum3D-Lite

端到端 3D 目标检测：多相机 2D 检测 + LiDAR 点云 → 360° 全景 3D BBox 回归。

> **项目状态: 进行中** — 后续将扩展更多功能（多目标跟踪 / 轨迹预测 / 占用网格 / 多数据集支持等）。

## 特性

- **360° 全向检测**: 6 相机 + LiDAR 覆盖车辆周围 40m 范围
- **端到端**: YOLO 2D 检测 → frustum 点云提取 → PointNet 3D 回归
- **轻量**: 50K 参数，单卡 GPU 推理
- **数据集可切换**: nuScenes / KITTI，通过 YAML 配置
- **传感器可配置**: 相机-LiDAR 内外参 YAML 驱动

## 效果 (360° 实时检测, 40 帧 scene, 50m)

> 左: 360° LiDAR 俯视图 (红=自身<1.5m). 右: 6 相机 YOLO 2D 检测框. 彩色框=3D BBox.

![demo](docs/images/demo.gif)

## 管线

```
多相机图像 (6×1600×900)                LIDAR_TOP (5-sweep)
        │                                      │
        ├─ YOLO → 2D bboxes                    ├─ 地面去除
        │                                      ├─ 自身点滤波 (<1.5m)
        │                                      │
        └────── frustum 裁剪 ──────────────────┘
                      │
              ROR 去噪 → DBSCAN 聚类
                      │
              采样 512 点 → PointNet3DDetector
                      │
              3D BBox: [cx, cy, cz, w, l, h, yaw]
```

## 模型

| 组件 | 规格 |
|------|------|
| 骨干 | PointNet (3→64→128 + MaxPool) |
| 融合特征 | 128(backbone) + 16(prior) + 16(centroid) + 16(extent) + 16(viewdir) + 16(face_cov) + 12(bbox_feat) = 220 |
| Center head | 220→192→96→3 |
| Size head | 220→96→3 |
| Yaw head | 220→96→2 |

## 指标 (nuScenes mini, 360°)

| 指标 | 值 |
|------|-----|
| Car center error | 0.26m |
| Car yaw error | 7.6° |
| Car size error | 0.15m |

## 快速开始

```bash
# 预处理
python scripts/tools/preprocess_phase3.py --nsweeps 5

# 训练
python scripts/train/train_phase3.py --config config/train.yaml --epochs 80

# 360° 可视化
python scripts/test/visualize_360.py
```

## 配置

| 文件 | 用途 |
|------|------|
| `config/train.yaml` | 训练参数、数据集路径、模型超参 |
| `config/sensor.yaml` | LiDAR/相机内外参、检测范围 |
| `config/phase3.yaml` | Phase 3 完整配置 (向后兼容) |

### 切换数据集

```yaml
# config/train.yaml
dataset:
  name: nuscenes        # nuscenes | kitti
  root: data/nuscenes   # 数据集路径
  version: v1.0-mini    # nuScenes 版本
```

## 项目结构

```
├── config/
│   ├── train.yaml              # 训练配置 (数据集/模型/损失)
│   ├── sensor.yaml             # 传感器标定配置
│   └── phase3.yaml             # Phase 3 完整配置
├── src/
│   ├── fusion.py               # PointNet3DDetector
│   ├── dataset_phase3.py       # 数据集 + frustum 管线
│   ├── inference.py            # 推理管线
│   ├── loss.py / metrics.py    # 损失 / 指标
│   ├── detector.py             # YOLO 检测器
│   └── ...
├── scripts/
│   ├── train/                  # 训练入口
│   │   ├── train_phase3.py
│   │   └── train_phase2.py
│   ├── test/                   # 测试/评估/可视化
│   │   ├── visualize_360.py
│   │   ├── visualize_scene.py
│   │   ├── visualize_infer.py
│   │   └── visualize_c2.py
│   └── tools/                  # 预处理/工具
│       └── preprocess_phase3.py
└── display/                    # 可视化输出 (gitignored)
```

## 坐标约定

LiDAR 帧: X=右, Y=前, Z=上. nuScenes 尺寸: `[width, length, height]`.
