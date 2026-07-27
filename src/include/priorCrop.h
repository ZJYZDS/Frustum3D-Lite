#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;

struct PriorSize {
    float w, l, h;
    PriorSize() = default;
    PriorSize(float width, float length, float height)
        : w(width), l(length), h(height) {}
};

// 类别先验尺寸: (w, l, h)
extern std::unordered_map<int, PriorSize> PriorMap;

// 小目标类别 (pedestrian, rider, motorcycle, bicycle)
bool isSmallClass(int class_id);

// 智能开关: 判断是否应启用 priorCrop
bool shouldEnablePriorCrop(
    const PointCloudPtr& frustum_cloud,
    int class_id,
    float depth_estimate,
    float bbox_height_px);

// 几何先验裁剪 (LiDAR 帧: X=右, Y=前, Z=上)
PointCloudPtr applyPriorCrop(
    PointCloudPtr frustum_cloud,
    int class_id,
    float bbox_cu, float bbox_cv,
    float bbox_h,
    float fx, float fy,
    float ppx, float ppy);
