/**
 * processFrustumWithDynamicParams — 动态参数聚类 + 采样
 *
 * 用物体先验尺寸 + 点云密度自适应计算 DBSCAN 参数,
 * 替代硬编码的 eps/min_samples, 然后在最大簇上采样到固定点数.
 *
 * 坐标系约定 (全程 LiDAR 帧):
 *   X = 右,  Y = 前 (深度方向),  Z = 上
 *
 * 调用注意:
 *   PriorSize 成员顺序为 (w, l, h), 传参时:
 *   processFrustumWithDynamicParams(cloud, prior.h, prior.w, prior.l)
 *   对应 header 参数 (object_height, object_width, object_length)
 */

#include "include/DynamicFrustum.h"
#include <random>

#define MAXTOLERANCE   0.15f   // 聚类最小容差 (m)
#define LEASTPTS       5       // 聚类最少点数下限
#define MAXPTS         1000    // 聚类最多点数上限 (防超大簇)
#define SAMPLEPTS      512     // 最终采样点数

PointCloudPtr processFrustumWithDynamicParams(
    const PointCloudPtr& frustum_cloud,
    float obj_h,              // 物体高度 (m)
    float objw,               // 物体宽度 (m)
    float objl,               // 物体长度 (m)
    float bbox_aspect_ratio,  // 2D bbox 宽/高比 (-1 = 不使用)
    bool isDebug)
{
    if (frustum_cloud->empty() || !frustum_cloud) {
        printf("[warn] processFrustum: input cloud empty/null\n");
        return nullptr;
    }

    if (isDebug)
        printf("[debug] input: %ld pts\n", frustum_cloud->size());

    // ══════════════════════════════════════════════════════════════════
    // 1. ROR 统计离群点剔除
    // ══════════════════════════════════════════════════════════════════
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(frustum_cloud);
    sor.setMeanK(20);
    sor.setStddevMulThresh(1.5);

    PointCloudPtr ror_cloud(new PointCloud);
    sor.filter(*ror_cloud);

    if (ror_cloud->size() < 10) {
        printf("[warn] after ROR: %ld pts (< 10)\n", ror_cloud->size());
        return nullptr;
    }

    if (isDebug)
        printf("[debug] after ROR: %ld pts\n", ror_cloud->size());

    // ══════════════════════════════════════════════════════════════════
    // 2. 计算平均深度 (LiDAR 帧: Y 轴 = 前向)
    // ══════════════════════════════════════════════════════════════════
    float avg_depth = 0.0f;
    int valid_pts = 0;
    for (const auto& p : ror_cloud->points) {
        float d = std::abs(p.y);                 // LiDAR Y = 前向深度
        if (std::isfinite(d) && d > 0.1f) {
            avg_depth += d;
            ++valid_pts;
        }
    }
    if (valid_pts == 0) return nullptr;
    avg_depth /= valid_pts;

    // ══════════════════════════════════════════════════════════════════
    // 3. 动态计算聚类参数
    // ══════════════════════════════════════════════════════════════════

    // 点间距 = 深度 × 角分辨率.  0.008 rad ≈ 0.46° (典型 LiDAR 水平角分辨率)
    const float angular_res = 0.008f;
    float point_spacing = avg_depth * angular_res;

    // 聚类容差: 物体最小尺寸 / 3, 但至少 MAXTOLERANCE
    float min_physics_dim = std::min(objw, objl);
    float physical_limit = min_physics_dim / 3.0f;
    float tolerance = std::min(physical_limit, point_spacing);
    tolerance = std::max(MAXTOLERANCE, tolerance);

    // 可见面积估算
    float min_side = std::min(objw, objl);
    float visible_area = obj_h * min_side;          // 可见面 ≈ 高 × 最小边

    // 可见比例: 从 bbox 宽高比推断看到了几面
    float visiable_factor = 0.5f;                   // 默认: 斜面
    if (bbox_aspect_ratio > 0.0f) {
        if (bbox_aspect_ratio > 1.2f) {              // 宽框 → 看到长边 + 侧面
            visiable_factor = 0.65f;
        } else if (bbox_aspect_ratio < 0.7f) {       // 窄框 → 看到短边 (正面/背面)
            visiable_factor = 0.35f;
        } else {                                     // 接近方形 → 正面
            visiable_factor = 0.75f;
        }
    }

    float obj_area = visiable_factor * visible_area;

    // 期望点数 = 可见面积 × 点密度
    // 点密度 = 1 / point_spacing² (pts/m²)
    float point_density = 1.0f / (point_spacing * point_spacing + 1e-5f);
    int ideal_points_num = static_cast<int>(obj_area * point_density);

    // min/max cluster size: 下限 = 期望点数的 25%, 上限 = 期望点数的 150%
    int min_num = std::max(10, static_cast<int>(ideal_points_num * 0.25f));
    int max_num = std::max(min_num + 1, static_cast<int>(ideal_points_num * 1.5f));

    // Clamp: floor to LEASTPTS, cap to MAXPTS
    min_num = (min_num < LEASTPTS) ? LEASTPTS : min_num;
    max_num = (max_num > MAXPTS) ? MAXPTS : max_num;

    if (isDebug)
        printf("[debug] depth=%.1fm spacing=%.3fm tolerance=%.3f "
               "ideal=%d min=%d max=%d\n",
               avg_depth, point_spacing, tolerance,
               ideal_points_num, min_num, max_num);

    // ══════════════════════════════════════════════════════════════════
    // 4. 欧氏聚类 → 取最大簇
    // ══════════════════════════════════════════════════════════════════
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(ror_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(tolerance);
    ec.setMinClusterSize(min_num);
    ec.setMaxClusterSize(max_num);
    ec.setSearchMethod(tree);
    ec.setInputCloud(ror_cloud);
    ec.extract(cluster_indices);

    // ── 聚类失败 → 回退: 直接用 ROR 后的全部点采样 ──
    if (cluster_indices.empty()) {
        if (isDebug)
            printf("[debug] no clusters → fallback to all ROR pts (%ld)\n",
                   ror_cloud->size());

        PointCloudPtr out(new PointCloud);
        size_t cur_pts = ror_cloud->size();
        if (cur_pts == 0) return nullptr;

        out->resize(SAMPLEPTS);
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, cur_pts - 1);
        for (size_t i = 0; i < SAMPLEPTS; ++i) {
            out->points[i] = ror_cloud->points[dis(gen)];
        }
        return out;
    }

    if (isDebug)
        printf("[debug] clusters found: %ld\n", cluster_indices.size());

    // 取最大簇
    auto max_cluster_it = std::max_element(
        cluster_indices.begin(), cluster_indices.end(),
        [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() < b.indices.size();
        });

    PointCloudPtr cluster_res(new PointCloud);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(ror_cloud);
    extract.setIndices(
        std::make_shared<const pcl::PointIndices>(*max_cluster_it));
    extract.filter(*cluster_res);

    // ══════════════════════════════════════════════════════════════════
    // 5. 采样到固定点数 SAMPLEPTS
    // ══════════════════════════════════════════════════════════════════
    PointCloudPtr out(new PointCloud);
    size_t cur_pts = cluster_res->size();

    if (cur_pts == SAMPLEPTS) {
        return cluster_res;
    } else if (cur_pts < SAMPLEPTS) {
        // 不足 → 随机重复填充
        out->resize(SAMPLEPTS);
        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, cur_pts - 1);
        for (size_t i = 0; i < SAMPLEPTS; ++i) {
            out->points[i] = cluster_res->points[dis(gen)];
        }
    } else {
        // 过多 → 随机降采样
        pcl::RandomSample<pcl::PointXYZ> rs;
        rs.setInputCloud(cluster_res);
        rs.setSample(static_cast<unsigned int>(SAMPLEPTS));
        rs.filter(*out);
    }

    return out;
}
