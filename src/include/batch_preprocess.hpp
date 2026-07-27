/**
 * batch_preprocess — OpenMP 并行化多 detection 预处理
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <string>

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;

struct FrustumJob {
    PointCloudPtr frustum_cloud;
    int    class_id;
    float  obj_h, obj_w, obj_l;
    float  bbox_aspect_ratio = -1.0f;

    // output
    PointCloudPtr output_cloud;
    bool    success = false;
    double  elapsed_ms = 0.0;
};

double processBatchSerial(std::vector<FrustumJob>& jobs);
double processBatchParallel(std::vector<FrustumJob>& jobs);
