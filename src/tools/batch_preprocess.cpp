/**
 * batch_preprocess — OpenMP 并行实现
 */

#include "include/batch_preprocess.hpp"
#include "DynamicFrustum.h"

#include <omp.h>
#include <chrono>
#include <cstdio>

// ── 串行实现 ────────────────────────────────────────────────────────────

double processBatchSerial(std::vector<FrustumJob>& jobs)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto& job : jobs) {
        auto t_start = std::chrono::high_resolution_clock::now();

        job.output_cloud = processFrustumWithDynamicParams(
            job.frustum_cloud,
            job.obj_h, job.obj_w, job.obj_l,
            job.bbox_aspect_ratio);

        auto t_end = std::chrono::high_resolution_clock::now();
        job.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        job.success = (job.output_cloud && !job.output_cloud->empty());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── 并行实现 ────────────────────────────────────────────────────────────

double processBatchParallel(std::vector<FrustumJob>& jobs)
{
    int n = static_cast<int>(jobs.size());
    if (n == 0) return 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        auto& job = jobs[i];
        auto t_start = std::chrono::high_resolution_clock::now();

        job.output_cloud = processFrustumWithDynamicParams(
            job.frustum_cloud,
            job.obj_h, job.obj_w, job.obj_l,
            job.bbox_aspect_ratio);

        auto t_end = std::chrono::high_resolution_clock::now();
        job.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        job.success = (job.output_cloud && !job.output_cloud->empty());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
