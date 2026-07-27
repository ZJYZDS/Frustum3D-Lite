"""Preprocessing pipeline: frustum cropping, denoising, clustering, multi-sweep aggregation."""

# nuScenes category → fine-tuned model class index
NUSCENES_CAT_TO_CLASS = {
    "vehicle.car": 2,
    "vehicle.truck": 3,
    "vehicle.bus": 4,
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7,
    "human.pedestrian": 0,
}

from pipeline.preprocess.frustum import (
    filter_points_by_frustum,
    filter_points_by_bbox_projection,
    compute_face_coverage,
    _compute_adaptive_margin,
)

from pipeline.preprocess.denoise import (
    remove_statistical_outliers,
    extract_largest_cluster,
    aggregate_sweeps,
    remove_ground_ransac,
)
