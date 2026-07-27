/**
 * applyPriorCrop + shouldEnablePriorCrop — 几何先验裁剪 (带智能开关)
 *
 * 在 DBSCAN 聚类之前, 利用类别先验尺寸 + 2D bbox 反投影估计物体 3D 位置,
 * 用一个宽松的 3D box 裁掉明显不属于该物体的背景点, 削减聚类输入规模.
 *
 * 坐标系约定 (全程 LiDAR 帧):
 *   X = 右,  Y = 前 (深度方向),  Z = 上
 *
 * 输入 frustum_cloud 来自 filter_points_by_frustum, 已经是 LiDAR 帧坐标.
 */

#include "include/priorCrop.h"

#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>
#include <unordered_map>
#include <cmath>
#include <cstdio>

using namespace std;

// ── 类别先验尺寸 ──────────────────────────────────────────────────────

unordered_map<int, PriorSize> PriorMap = {
    {0, {0.70f, 0.70f, 1.70f}},  {1, {0.70f, 0.70f, 1.70f}},
    {2, {1.90f, 4.60f, 1.50f}},  {3, {2.50f, 6.50f, 2.80f}},
    {4, {2.80f, 10.5f, 3.20f}},  {5, {3.00f, 15.0f, 4.00f}},
    {6, {0.70f, 2.00f, 1.50f}},  {7, {0.60f, 1.80f, 1.30f}},
    {8, {0.30f, 0.30f, 0.80f}},  {9, {0.30f, 0.30f, 0.80f}},
};

// ── 小目标类别 ──────────────────────────────────────────────────────

bool isSmallClass(int class_id) {
    return class_id == 0 || class_id == 1 || class_id == 6 || class_id == 7;
}

// ── 智能开关: 判断是否应该启用裁剪 ─────────────────────────────────────

bool shouldEnablePriorCrop(
    const PointCloudPtr& frustum_cloud,
    int class_id,
    float depth_estimate,   // 从 2D bbox 估算的深度 (m)
    float bbox_height_px)   // 2D bbox 像素高度
{
    // 1. 点数太少 → 裁剪无意义 (本身就是稀疏目标)
    if (!frustum_cloud || frustum_cloud->size() < 50)
        return false;

    // 2. 远距离 (>25m) → 视锥已很小, 裁剪收益为 0
    if (depth_estimate > 25.0f)
        return false;

    // 3. 小目标 + 中远距离 (>15m) → 先验尺寸小, 深度估计误差大, 容易误杀
    if (isSmallClass(class_id) && depth_estimate > 15.0f)
        return false;

    // 4. 2D bbox 太小 (< 20px) → 深度估计不稳定
    if (bbox_height_px < 20.0f)
        return false;

    // 5. 类别不在先验表中 → 无法裁剪
    if (PriorMap.find(class_id) == PriorMap.end())
        return false;

    return true;
}

// ── 主函数 ─────────────────────────────────────────────────────────────

PointCloudPtr applyPriorCrop(
    PointCloudPtr frustum_cloud,   // LiDAR 帧点云 (X=右, Y=前, Z=上)
    int class_id,
    float bbox_cu, float bbox_cv,  // 2D bbox 中心 (像素坐标)
    float bbox_h,                  // 2D bbox 像素高度
    float fx, float fy,            // 相机内参
    float ppx, float ppy)
{
    if (frustum_cloud->empty() || !frustum_cloud) {
        printf("[priorCrop] frustum_cloud is empty or nullptr\n");
        return frustum_cloud;
    }

    if (PriorMap.find(class_id) == PriorMap.end()) {
        return frustum_cloud;   // 无先验尺寸, 跳过
    }

    float real_w = PriorMap[class_id].w;
    float real_l = PriorMap[class_id].l;
    float real_h = PriorMap[class_id].h;

    // ── 1. 从 2D bbox 像素高度估计深度 (沿 LiDAR Y 轴, 前向) ──
    if (bbox_h < 1.0f) return frustum_cloud;   // bbox 太小, 估计不可靠
    float depth = (fy * real_h) / bbox_h;       // 深度 = f * 实际高度 / 像素高度
    if (depth < 1.0f || depth > 55.0f) return frustum_cloud;

    // ── 2. 反投影 → 3D 中心 (LiDAR 帧) ──
    float cx = (bbox_cu - ppx) * depth / fx;       // LiDAR X (右)
    float cy_cam = (bbox_cv - ppy) * depth / fy;    // Camera Y (下)
    float cz = -cy_cam;                              // LiDAR Z (上)

    // ── 3. 动态 margin ──
    // 基础值: 深度方向最不可靠, 给最大余量
    float margin_depth = 2.0f;
    float margin_lateral = real_w * 0.5f;
    float margin_vertical = real_h * 1.0f;

    // 规则 A: 小目标 (ped/rider/moto/bike) — 先验尺寸小, 放宽 margin
    if (isSmallClass(class_id)) {
        margin_depth   = 4.0f;    // 前向 ±4m
        margin_lateral = 1.0f;    // 侧向 ±1m (而非基于 real_w)
        margin_vertical = 2.0f;   // 垂直 ±2m (而非基于 real_h)
    }

    // 规则 B: 深度 > 20m — 深度估计误差随距离增大, 进一步放宽
    if (depth > 20.0f) {
        margin_depth   = 6.0f;
        margin_lateral = std::max(margin_lateral, 1.5f);
        margin_vertical = std::max(margin_vertical, 2.5f);
    }

    // ── 4. 3D CropBox (LiDAR 帧) ──
    Eigen::Vector4f min_pt(
        cx - real_w / 2 - margin_lateral,      // X (右)
        depth - margin_depth,                   // Y (前)  ← 深度方向
        cz - real_h / 2 - margin_vertical,     // Z (上)
        1.0f
    );
    Eigen::Vector4f max_pt(
        cx + real_w / 2 + margin_lateral,
        depth + margin_depth,
        cz + real_h / 2 + margin_vertical,
        1.0f
    );

    // ── 5. 裁剪 ──
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(frustum_cloud);
    crop.setMin(min_pt);
    crop.setMax(max_pt);

    PointCloudPtr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    crop.filter(*cropped_cloud);

    // ── 6. 第一次裁剪失败 → 用点云质心修正深度, 重试 ──
    if (cropped_cloud->size() < 5) {
        printf("[priorCrop] 1st crop=%ld pts < 5 → centroid retry\n",
               cropped_cloud->size());

        // 计算 frustum 质心, 用质心的 Y (前向) 作为修正深度
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*frustum_cloud, centroid);
        float depth_corrected = centroid[1];  // LiDAR Y = 前向深度

        // 只信任合理范围内的修正
        if (depth_corrected > 1.0f && depth_corrected < 55.0f) {
            // 修正后重新反投影中心
            float cx2 = (bbox_cu - ppx) * depth_corrected / fx;
            float cy2_cam = (bbox_cv - ppy) * depth_corrected / fy;
            float cz2 = -cy2_cam;

            // 第二次裁剪: margin 放宽 50%
            float md2 = margin_depth * 1.5f;
            float ml2 = margin_lateral * 1.5f;
            float mv2 = margin_vertical * 1.5f;

            Eigen::Vector4f min_pt2(
                cx2 - real_w/2 - ml2,
                depth_corrected - md2,
                cz2 - real_h/2 - mv2, 1.0f);
            Eigen::Vector4f max_pt2(
                cx2 + real_w/2 + ml2,
                depth_corrected + md2,
                cz2 + real_h/2 + mv2, 1.0f);

            pcl::CropBox<pcl::PointXYZ> crop2;
            crop2.setInputCloud(frustum_cloud);
            crop2.setMin(min_pt2);
            crop2.setMax(max_pt2);
            PointCloudPtr retry_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            crop2.filter(*retry_cloud);

            printf("[priorCrop] retry: depth %.1f→%.1fm  %ld pts\n",
                   depth, depth_corrected, retry_cloud->size());

            if (retry_cloud->size() >= 5)
                return retry_cloud;
        }
        printf("[priorCrop] retry also failed → fallback\n");
        return frustum_cloud;
    }

    printf("[priorCrop] %ld → %ld pts (depth=%.1fm, margin_depth=%.1fm)\n",
           frustum_cloud->size(), cropped_cloud->size(), depth, margin_depth);

    return cropped_cloud;
}
