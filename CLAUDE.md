# Cross-Modal 3D BBox Refinement (PointNet++)

## 项目概述
CAM_FRONT + LIDAR_TOP → YOLO 2D检测 → 3D bbox回归.
- **Phase 2** (主线): C2/C3 残差回归 | **Phase 3** (开发中): DualHeadPointNet 绝对回归 157K

## 关键文件
- `src/fusion.py` — C1/C2/C3 + Phase 3 DualHeadPointNet
- `src/model.py` — PointNet++ FPS/Ball Query/Set Abstraction
- `src/dataset_phase2.py` — Phase 2: YOLO→crop→残差 | `src/dataset_phase3.py` — Phase 3: 多帧聚合+地面去除+视锥+绝对回归
- `src/dataset_phase1.py` — LiDARProjector (复用) | `src/detector.py` — YOLO ONNX
- `src/loss.py` — BboxRefinementLoss(残差) + bbox_loss(绝对) | `src/metrics.py` — 两套指标
- `src/ground_removal.py` — RANSAC 地面去除 | `src/init_estimator.py` — 2D→3D 初始化
- `scripts/train_phase2.py` / `train_phase3.py` | `config/phase2.yaml` / `phase3.yaml`
- `src/new_model_arch.md` — Phase 3 架构设计文档

## 坐标帧 (每次改数据管线必须检查!)

### 帧定义与变换
```
global ←→ ego ←→ LiDAR sensor      nuScenes: size=(宽,长,高), yaw=0°=x轴前
              ←→ camera
```
- global→ego: `R_ego^T @ (pt - t_ego)` | ego→LiDAR: `R_lidar^T @ (pt - t_lidar)`
- LiDAR→camera: `R_cam^T @ (R_lidar@pt + t_lidar - t_cam)` (即 T_lidar2cam)

### Phase 3 管线 (每步均在 LiDAR 帧)
```
aggregate_sweeps:  sensor→ego→ref_ego→LiDAR    (最后一步是关键!)
remove_ground:     纯几何, 不改变帧
frustum_filter:    T_lidar2cam 期望 LiDAR 输入
ROR+DBSCAN:        纯几何
_global_to_lidar:  global→ego→LiDAR             (标签)
augmentation:      pts+center 同旋转, yaw 同步
```

### 检查清单
- [ ] 点云在哪个帧? (.bin=LiDAR sensor; 多帧聚合后需补 ego→LiDAR)
- [ ] GT 标签在哪个帧? (需经 `_global_to_lidar`)
- [ ] yaw 在哪个帧? (需经 `_quaternion_to_yaw_lidar`)
- [ ] T_lidar2cam 与点云帧匹配?
- [ ] 增强旋转同步应用到点云+标签?

### 已知陷阱
1. **aggregate_sweeps 缺 ego→LiDAR (已修复)**: 输出 ego 帧 vs 标签 LiDAR 帧, 差 ~1-2m
2. **ground_removal._multisweep_lidar 输出 ego 帧**: Phase 3 未使用, 但需注意
3. **Phase 2 _build_target 坐标不匹配 (已修复)**: obj_points(LiDAR) - noisy_center(Ego)

## 训练

Phase 2: `python scripts/train_phase2.py --model_type c2_3d`
- nuScenes v1.0-mini, 8 train+2 val, 残差截断 center≤2m/size≤1m/yaw≤20°
- loss 权重 center=1.0/size=2.0/yaw=1.5

Phase 3: `python scripts/train_phase3.py`
- 绝对回归 [cx,cy,cz,w,h,l,sin,cos], nsweeps=10, 512点, max_dist=15m
- RANSAC地面(0.25m)+视锥+ROR+DBSCAN, Z轴旋转增强
