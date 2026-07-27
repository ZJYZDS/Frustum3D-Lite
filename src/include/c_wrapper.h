/**
 * C-linkage wrapper for C++ preprocessing functions.
 * Callable from Python via ctypes.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Process a frustum point cloud through ROR + dynamic clustering + sampling.
 *
 * @param pts_in      Input points: N*3 floats (x,y,z), LiDAR frame
 * @param n_in        Number of input points
 * @param obj_h       Object height (m) — from PriorSize
 * @param obj_w       Object width (m)
 * @param obj_l       Object length (m)
 * @param bbox_ar     2D bbox aspect ratio (w/h), -1 = unknown
 * @param pts_out     Output buffer: max_out*3 floats (pre-allocated)
 * @param max_out     Capacity of output buffer (recommend 512)
 * @param n_out       [out] Actual number of output points written
 * @return            0 on success, -1 on error
 */
int cpp_process_frustum(
    const float* pts_in, int n_in,
    float obj_h, float obj_w, float obj_l, float bbox_ar,
    float* pts_out, int max_out, int* n_out);

/**
 * Apply prior-based geometric crop to a frustum point cloud.
 *
 * @return  number of output points (>0 on success, =0 means fallback=use input)
 */
int cpp_apply_prior_crop(
    const float* pts_in, int n_in,
    int class_id,
    float bbox_cu, float bbox_cv, float bbox_h,
    float fx, float fy, float ppx, float ppy,
    float* pts_out, int max_out);

/**
 * Smart switch: should we enable priorCrop for this frustum?
 * @return 1 = enable, 0 = skip
 */
int cpp_should_enable_prior_crop(
    int n_points, int class_id,
    float depth_estimate, float bbox_height_px);

#ifdef __cplusplus
}
#endif
