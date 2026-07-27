"""360° panorama: stitch 6 cameras into one image for single-pass YOLO detection."""

import cv2
import numpy as np


def stitch_panorama(images, target_height=480):
    """Stitch 6 camera images into a horizontal panorama.

    All cameras have the same resolution (1600x900 for nuScenes).
    Simply resize to target_height and concatenate horizontally.

    Args:
        images: dict {camera_name: BGR image}
        target_height: int, resize height

    Returns:
        panorama: (H, W, 3) stitched image
        x_offsets: list of (camera_name, x_start, x_end) mapping
    """
    order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    strips = []
    x_offsets = []
    x = 0

    for cam in order:
        if cam not in images:
            continue
        img = images[cam]
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height))
        strips.append(resized)
        x_offsets.append((cam, x, x + new_w, scale))
        x += new_w

    panorama = np.hstack(strips) if strips else None
    return panorama, x_offsets


def split_detections(dets, x_offsets):
    """Map panorama detections back to original cameras.

    Args:
        dets: list[dict] YOLO detections with 'bbox' in panorama pixel coords
        x_offsets: list of (camera_name, x_start, x_end, scale)

    Returns:
        dict {camera_name: list[dict]} per-camera detections
    """
    cam_dets = {cam: [] for cam, _, _, _ in x_offsets}

    for det in dets:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2

        for cam, xs, xe, scale in x_offsets:
            if xs <= cx < xe:
                # Remap bbox to original camera coordinates
                new_bbox = np.array([
                    (x1 - xs) / scale,
                    y1 / scale,
                    (x2 - xs) / scale,
                    y2 / scale,
                ])
                new_det = dict(det)
                new_det['bbox'] = new_bbox
                new_det['camera'] = cam
                cam_dets[cam].append(new_det)
                break

    return cam_dets
