/**
 * C-linkage wrapper — bridges Python ctypes ↔ C++ PCL preprocessing.
 */

#include "../include/c_wrapper.h"
#include "../include/priorCrop.h"
#include "../include/DynamicFrustum.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstring>
#include <cstdio>

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
using PointCloudPtr = PointCloud::Ptr;

// ── Helper: raw float* → PCL PointCloud ─────────────────────────────────

static PointCloudPtr arrayToCloud(const float* pts, int n) {
    auto cloud = std::make_shared<PointCloud>();
    cloud->resize(n);
    for (int i = 0; i < n; ++i) {
        cloud->points[i].x = pts[i * 3];
        cloud->points[i].y = pts[i * 3 + 1];
        cloud->points[i].z = pts[i * 3 + 2];
    }
    return cloud;
}

static int cloudToArray(const PointCloudPtr& cloud, float* out, int max_out) {
    int n = std::min((int)cloud->size(), max_out);
    for (int i = 0; i < n; ++i) {
        out[i * 3]     = cloud->points[i].x;
        out[i * 3 + 1] = cloud->points[i].y;
        out[i * 3 + 2] = cloud->points[i].z;
    }
    return n;
}

// ── Public C API ────────────────────────────────────────────────────────

int cpp_process_frustum(
    const float* pts_in, int n_in,
    float obj_h, float obj_w, float obj_l, float bbox_ar,
    float* pts_out, int max_out, int* n_out)
{
    if (!pts_in || n_in < 10 || !pts_out || !n_out) return -1;

    try {
        auto cloud = arrayToCloud(pts_in, n_in);
        auto result = processFrustumWithDynamicParams(
            cloud, obj_h, obj_w, obj_l, bbox_ar);

        if (!result || result->empty()) {
            *n_out = 0;
            return 0;  // clustering failed, no points
        }

        *n_out = cloudToArray(result, pts_out, max_out);
        return 0;
    } catch (...) {
        return -1;
    }
}

int cpp_apply_prior_crop(
    const float* pts_in, int n_in,
    int class_id,
    float bbox_cu, float bbox_cv, float bbox_h,
    float fx, float fy, float ppx, float ppy,
    float* pts_out, int max_out)
{
    if (!pts_in || n_in < 5 || !pts_out) return 0;

    try {
        auto cloud = arrayToCloud(pts_in, n_in);

        // Compute depth estimate for smart switch
        auto it = PriorMap.find(class_id);
        float depth_est = 0;
        if (it != PriorMap.end() && bbox_h >= 1.0f)
            depth_est = (fy * it->second.h) / bbox_h;

        // Smart switch
        if (!shouldEnablePriorCrop(cloud, class_id, depth_est, bbox_h)) {
            // Skip: copy input as-is
            int n = std::min(n_in, max_out);
            std::memcpy(pts_out, pts_in, n * 3 * sizeof(float));
            return n;
        }

        auto result = applyPriorCrop(cloud, class_id,
                                     bbox_cu, bbox_cv, bbox_h,
                                     fx, fy, ppx, ppy);
        return cloudToArray(result, pts_out, max_out);
    } catch (...) {
        return 0;
    }
}

int cpp_should_enable_prior_crop(
    int n_points, int class_id,
    float depth_estimate, float bbox_height_px)
{
    // Create minimal point cloud just for the size check
    auto cloud = std::make_shared<PointCloud>();
    cloud->resize(n_points);  // dummy, only size matters for the check
    return shouldEnablePriorCrop(cloud, class_id, depth_estimate, bbox_height_px) ? 1 : 0;
}
