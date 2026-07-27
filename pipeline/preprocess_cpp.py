"""
Python ctypes bridge → C++ preprocessing (ROR + clustering + priorCrop + sampling).

Usage:
    from pipeline.preprocess_cpp import CppPreprocessor

    cpp = CppPreprocessor()
    # Single-detection full pipeline
    result = cpp.process_frustum(frustum_pts, obj_h=1.5, obj_w=1.9, obj_l=4.6)
    # Or with priorCrop
    result = cpp.process_with_prior_crop(frustum_pts, class_id=2, ...)
"""

import ctypes
import numpy as np
import os
from pathlib import Path

# Find the shared library
_LIB_PATH = Path(__file__).resolve().parent.parent / "build" / "lib" / "libfrustum_preprocess.so"

if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"C++ shared library not found at {_LIB_PATH}. "
        f"Run: cd build && cmake .. && make frustum_preprocess")

_lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))

# ── Function signatures ──────────────────────────────────────────────────

_lib.cpp_process_frustum.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # pts_in
    ctypes.c_int,                     # n_in
    ctypes.c_float, ctypes.c_float, ctypes.c_float,  # obj_h, w, l
    ctypes.c_float,                   # bbox_aspect_ratio
    ctypes.POINTER(ctypes.c_float),  # pts_out
    ctypes.c_int,                     # max_out
    ctypes.POINTER(ctypes.c_int),    # n_out
]
_lib.cpp_process_frustum.restype = ctypes.c_int

_lib.cpp_apply_prior_crop.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
_lib.cpp_apply_prior_crop.restype = ctypes.c_int

_lib.cpp_should_enable_prior_crop.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float,
]
_lib.cpp_should_enable_prior_crop.restype = ctypes.c_int

# ── Prior size table (must match C++ PriorMap) ──────────────────────────

PRIOR_SIZE = {
    0:  (0.70, 0.70, 1.70),   # pedestrian (w, l, h)
    1:  (0.70, 0.70, 1.70),   # rider
    2:  (1.90, 4.60, 1.50),   # car
    3:  (2.50, 6.50, 2.80),   # truck
    4:  (2.80, 10.5, 3.20),   # bus
    5:  (3.00, 15.0, 4.00),   # train
    6:  (0.70, 2.00, 1.50),   # motorcycle
    7:  (0.60, 1.80, 1.30),   # bicycle
    8:  (0.30, 0.30, 0.80),   # traffic light
    9:  (0.30, 0.30, 0.80),   # traffic sign
}

MAX_OUT = 2048  # max output points buffer size


class CppPreprocessor:
    """C++ preprocessing wrapper — replaces Python ROR + DBSCAN + sampling."""

    def __init__(self):
        self._out_buf = np.zeros(MAX_OUT * 3, dtype=np.float32)
        self._n_out = ctypes.c_int(0)

    def process_frustum(self, pts, class_id=2, bbox_aspect_ratio=-1.0):
        """Full C++ pipeline: ROR → dynamic clustering → sampling to 512 pts.

        Args:
            pts: (N, 3) float32 numpy array, LiDAR frame
            class_id: int
            bbox_aspect_ratio: 2D bbox w/h (-1 = unknown)

        Returns:
            (M, 3) float32 numpy array (M <= 512), or None on failure
        """
        if len(pts) < 10:
            return None

        w, l, h = PRIOR_SIZE.get(class_id, PRIOR_SIZE[2])

        pts_c = np.ascontiguousarray(pts.astype(np.float32))
        n_in = ctypes.c_int(len(pts_c))

        rc = _lib.cpp_process_frustum(
            pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_in,
            ctypes.c_float(h), ctypes.c_float(w), ctypes.c_float(l),
            ctypes.c_float(bbox_aspect_ratio),
            self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(MAX_OUT),
            ctypes.byref(self._n_out),
        )

        if rc != 0 or self._n_out.value <= 0:
            return None

        return self._out_buf[:self._n_out.value * 3].reshape(-1, 3).copy()

    def process_with_prior_crop(self, pts, class_id,
                                 bbox_cu, bbox_cv, bbox_h,
                                 fx, fy, ppx, ppy,
                                 bbox_aspect_ratio=-1.0):
        """Apply priorCrop first, then C++ full pipeline.

        Args:
            pts: (N, 3) frustum points, LiDAR frame
            ...camera params...
            bbox_aspect_ratio: for dynamic clustering

        Returns:
            (M, 3) float32 numpy array (M <= 512), or None
        """
        if len(pts) < 5:
            return None

        pts_c = np.ascontiguousarray(pts.astype(np.float32))
        n_in = ctypes.c_int(len(pts_c))

        # Step 1: priorCrop
        n_cropped = _lib.cpp_apply_prior_crop(
            pts_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_in,
            ctypes.c_int(class_id),
            ctypes.c_float(bbox_cu), ctypes.c_float(bbox_cv),
            ctypes.c_float(bbox_h),
            ctypes.c_float(fx), ctypes.c_float(fy),
            ctypes.c_float(ppx), ctypes.c_float(ppy),
            self._out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(MAX_OUT),
        )

        if n_cropped < 5:
            return None  # priorCrop fallback with too few points

        cropped_pts = self._out_buf[:n_cropped * 3].copy()

        # Step 2: dynamic clustering + sampling
        return self.process_frustum(
            cropped_pts.reshape(-1, 3), class_id, bbox_aspect_ratio)

    def should_crop(self, n_points, class_id, depth_est, bbox_h):
        """Check if priorCrop should be applied."""
        return bool(_lib.cpp_should_enable_prior_crop(
            ctypes.c_int(n_points), ctypes.c_int(class_id),
            ctypes.c_float(depth_est), ctypes.c_float(bbox_h)))
