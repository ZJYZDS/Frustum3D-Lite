#ifndef DYNAMICFRUSTUM_H
#define DYNAMICFRUSTUM_H


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <cmath>
#include <algorithm>
#include <memory>
#include <stdio.h>

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

/**
 * @brief 动态聚类并采样最大簇，利用物体尺寸先验自适应调整参数
 * @param frustum_cloud     输入视锥点云（传感器坐标系下）
 * @param object_height     物体高度 (m)
 * @param object_width      物体宽度 (m)
 * @param object_length     物体长度 (m)
 * @param bbox_aspect_ratio 2D检测框宽高比 (width/height)，用于判断可见面数，默认-1表示不使用
 * @param isDebug           是否打印调试信息，默认false
 * @return pcl::PointCloud<pcl::PointXYZ>::Ptr 处理后的最大簇点云（可能已降采样），若失败返回nullptr
 */
PointCloudPtr processFrustumWithDynamicParams(
    const PointCloudPtr& frustum_cloud,
    float object_height,
    float object_width,
    float object_length,
    float bbox_aspect_ratio = -1.0f,
    bool isDebug = false);



#endif