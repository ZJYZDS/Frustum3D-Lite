"""
Performance profiling utilities for Frustum3D-Lite inference pipeline.

Provides:
  - StageTimer: context manager for per-stage wall-clock timing
  - PipelineProfiler: collects, accumulates, and reports per-frame timing breakdowns
"""

import time
import math
from contextlib import contextmanager


class PipelineProfiler:
    """Collect per-stage timing across multiple frames, report breakdown.

    Usage:
        profiler = PipelineProfiler()

        with profiler.stage("load"):
            pts = load_point_cloud(...)

        with profiler.stage("filter"):
            pts = filter_ground(pts)

        # ... more stages ...

        profiler.print_breakdown()   # per-frame summary
    """

    def __init__(self):
        self._records = []          # list[dict] — per-frame stage→ms maps
        self._current_frame = {}    # stage_name → elapsed_ms
        self._frame_count = 0

    @contextmanager
    def stage(self, name: str):
        """Context manager that times a named stage.

        Usage:
            with profiler.stage("load"):
                pts = aggregate_sweeps(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0  # ms
            self._current_frame[name] = elapsed

    def new_frame(self):
        """Commit current frame's timings and start a fresh frame."""
        if self._current_frame:
            self._records.append(self._current_frame)
            self._current_frame = {}
            self._frame_count += 1

    @property
    def frame_count(self):
        return self._frame_count

    def last_frame(self):
        """Return the most recent frame's stage→ms dict (before new_frame)."""
        return dict(self._current_frame)

    def summary(self):
        """Return aggregated stats across all recorded frames.

        Returns:
            dict: {stage_name: {'mean': ms, 'min': ms, 'max': ms, 'std': ms, 'pct': %}}
        """
        if not self._records and not self._current_frame:
            return {}

        all_records = list(self._records)
        if self._current_frame:
            all_records.append(self._current_frame)

        # Gather all stage names
        stage_names = []
        for r in all_records:
            for k in r:
                if k not in stage_names:
                    stage_names.append(k)

        stats = {}
        for name in stage_names:
            values = [r[name] for r in all_records if name in r]
            if not values:
                continue
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            stats[name] = {
                'mean': mean_val,
                'min': min(values),
                'max': max(values),
                'std': math.sqrt(variance),
                'count': len(values),
            }

        # Compute percentage of total (using mean values)
        total_mean = sum(s['mean'] for s in stats.values())
        if total_mean > 0:
            for name in stats:
                stats[name]['pct'] = stats[name]['mean'] / total_mean * 100.0

        return stats

    def print_breakdown(self, title="Frustum3D-Lite Per-frame Breakdown"):
        """Print a formatted per-stage timing report for the last frame."""
        frame = self._current_frame if self._current_frame else (
            self._records[-1] if self._records else {}
        )
        if not frame:
            print("[Profiler] No timing data recorded.")
            return

        total = sum(frame.values())
        if total <= 0:
            return

        fps = 1000.0 / total if total > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        # Sort by insertion order (preserved in Python 3.7+)
        for name, ms in frame.items():
            pct = ms / total * 100.0
            bar = '█' * int(pct / 2)  # crude bar: 2% per block
            print(f"  {name:<25s}: {ms:7.1f} ms ({pct:4.1f}%)  {bar}")
        print(f"  {'─'*54}")
        print(f"  {'TOTAL':25s}: {total:7.1f} ms (100.0%)")
        print(f"  {'FPS':25s}: {fps:7.1f}")
        print(f"{'='*60}\n")

    def print_summary(self, title="Frustum3D-Lite Summary (all frames)"):
        """Print aggregated stats across all recorded frames."""
        stats = self.summary()
        if not stats:
            print("[Profiler] No timing data recorded.")
            return

        total_mean = sum(s['mean'] for s in stats.values())

        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"  ({self._frame_count} frames)")
        print(f"{'='*70}")
        print(f"  {'Stage':<25s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'%':>6s}")
        print(f"  {'─'*60}")
        for name, s in stats.items():
            print(f"  {name:<25s} {s['mean']:7.1f}ms {s['min']:7.1f}ms "
                  f"{s['max']:7.1f}ms {s['pct']:5.1f}%")
        print(f"  {'─'*60}")
        print(f"  {'TOTAL (mean)':25s} {total_mean:7.1f}ms")
        fps = 1000.0 / total_mean if total_mean > 0 else 0.0
        print(f"  {'FPS':25s} {fps:7.1f}")
        print(f"{'='*70}\n")
