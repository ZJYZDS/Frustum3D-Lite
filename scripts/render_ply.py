"""PLY → PNG 渲染: 为 README 展示生成点云 + 3D bbox 截图."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = "display"
VIEWS = {
    "persp": dict(elev=25, azim=-60),
    "top": dict(elev=90, azim=0),
    "front": dict(elev=0, azim=-90),
}


def read_ply(path):
    """简易 PLY reader: 只支持 ascii, vertex-only."""
    pts, colors = [], []
    in_header, n_verts = True, 0
    vert_count = 0
    with open(path) as f:
        for line in f:
            if in_header:
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("end_header"):
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            xyz = list(map(float, parts[:3]))
            rgb = list(map(int, parts[3:6]))
            pts.append(xyz)
            colors.append(rgb)
            vert_count += 1
            if vert_count >= n_verts:
                break
    return np.array(pts, dtype=np.float32), np.array(colors, dtype=np.float32) / 255.0


def render_ply(ply_path, out_path, elev=25, azim=-60):
    pts, colors = read_ply(ply_path)
    if len(pts) == 0:
        return

    # 区分 bbox 边线 (红/绿/蓝) 与背景点云
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    is_red = (r > 0.9) & (g < 0.3) & (b < 0.3)
    is_green = (r < 0.1) & (g > 0.8) & (b < 0.1)
    is_blue = (r < 0.5) & (g < 0.7) & (b > 0.9)
    bbox_mask = is_red | is_green | is_blue
    bg_mask = ~bbox_mask

    # bg 降采样到 15000 点
    bg_idx = np.where(bg_mask)[0]
    if len(bg_idx) > 15000:
        bg_idx = np.random.RandomState(42).choice(bg_idx, 15000, replace=False)
    keep = np.concatenate([bg_idx, np.where(bbox_mask)[0]])
    pts, colors = pts[keep], colors[keep]

    # 重新计算 masks (降采样后)
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    bbox_mask = (r > 0.9) & (g < 0.3) & (b < 0.3)
    bbox_mask |= (r < 0.1) & (g > 0.8) & (b < 0.1)
    bbox_mask |= (r < 0.5) & (g < 0.7) & (b > 0.9)
    bg_mask = ~bbox_mask

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    # 背景点云 — 小点半透明
    ax.scatter(pts[bg_mask, 0], pts[bg_mask, 1], pts[bg_mask, 2],
               c=colors[bg_mask], s=0.2, alpha=0.5)
    # bbox 边线 — 大点不透明
    ax.scatter(pts[bbox_mask, 0], pts[bbox_mask, 1], pts[bbox_mask, 2],
               c=colors[bbox_mask], s=4.0, alpha=1.0)

    # 自动缩放: 以 bbox 区域为中心, 紧贴物体
    if bbox_mask.any():
        mid = pts[bbox_mask].mean(axis=0)
        span = max(np.ptp(pts[bbox_mask][:, 0]), np.ptp(pts[bbox_mask][:, 1]),
                   np.ptp(pts[bbox_mask][:, 2])) / 2 + 8
    else:
        mid = pts.mean(axis=0)
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])) / 2 + 5
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.endswith(".ply"):
            continue
        ply_path = os.path.join(OUT_DIR, fname)
        base = fname.replace(".ply", "")
        for vname, vparams in VIEWS.items():
            out_path = os.path.join(OUT_DIR, f"{base}_{vname}.png")
            render_ply(ply_path, out_path, **vparams)


if __name__ == "__main__":
    main()
