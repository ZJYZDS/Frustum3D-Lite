"""
YOLO ONNX 轻量推理模块.

提供两个类:
  - YOLOSegONNX:   分割模型 (YOLOv8-seg), 输出 bbox + mask. Phase 2 未使用.
  - YOLODetectONNX: 检测模型 (YOLO26s), 仅输出 bbox. Phase 2 使用此类.

Phase 2 选择 YOLODetectONNX 的原因:
  1. 推理更快 (37MB vs 55MB+), 无 mask head 开销
  2. bbox 内的 LiDAR 点云已足够定位, mask 提纯收益有限
  3. 输出格式简单: (1, N, 6) = [x1,y1,x2,y2,conf,cls], NMS 已内置
"""

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# 微调模型 10 类
FINETUNE_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle", "traffic light", "traffic sign",
]

# 3D 检测: YOLO 微调模型可检测的全部障碍物类别
# is_person: True → yaw 不参与训练 (点云几何无方向信息)
OBSTACLE_CLASSES = {
    0:  ("pedestrian",     True),   # 行人
    1:  ("rider",          True),   # 骑行者
    2:  ("car",            False),
    3:  ("truck",          False),
    4:  ("bus",            False),
    5:  ("train",          False),  # 火车 (nuScenes 稀有)
    6:  ("motorcycle",     False),
    7:  ("bicycle",        False),
    8:  ("traffic light",  False),  # 交通灯
    9:  ("traffic sign",   False),  # 交通标志
}
OBSTACLE_CLASS_IDS = set(OBSTACLE_CLASSES.keys())


# ==============================================================================
# YOLODetectONNX: Phase 2 使用的检测推理 (主要使用的类)
# ==============================================================================

class YOLODetectONNX:
    """非分割 YOLO 模型的 ONNX 推理 (YOLO26s, YOLOv8-detect 等).

    输出格式: (1, N, 6) = [x1, y1, x2, y2, confidence, class_id]
    NMS 已在导出 ONNX 时内置 (end2end export), 所以后处理极简:
      1. 置信度过滤
      2. bbox 坐标从 model space → original image space
      3. clip 到图像边界
    """

    def __init__(self, onnx_path, conf_thresh=0.5, imgsz=640):
        self.conf = conf_thresh
        self.imgsz = imgsz

        # ONNX Runtime: 优先 CUDA, 回退 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]

        print(f"[YOLODetectONNX] Loaded {onnx_path}")
        print(f"  Input: {self.input_name} {input_shape}")
        for o in self.session.get_outputs():
            print(f"  Output: {o.name} {o.shape}")

    def preprocess(self, img_bgr):
        """Letterbox 预处理: 保持宽高比缩放 + 居中填充到正方形.

        必须和导出 ONNX 时的预处理一致, 否则 bbox 坐标映射会出错.

        Returns:
            tensor: (1, 3, H_in, W_in) float32 [0,1]
            (ratio_w, ratio_h): 缩放因子, 用于将 bbox 坐标映射回原始图像
            (pad_w, pad_h):     填充偏移, 同上
        """
        h0, w0 = img_bgr.shape[:2]

        # 缩放因子 (保持宽高比, 不拉伸)
        r = min(self.input_w / w0, self.input_h / h0)
        new_w, new_h = int(w0 * r), int(h0 * r)
        ratio_w, ratio_h = w0 / new_w, h0 / new_h

        # 缩放
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 居中填充 (letterbox)
        dw = (self.input_w - new_w) / 2
        dh = (self.input_h - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # BGR→RGB, HWC→CHW, uint8→float32, [0,255]→[0,1]
        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, (ratio_w, ratio_h), (left, top)

    def predict(self, img_bgr):
        """对单张图像运行检测.

        Args:
            img_bgr: (H, W, 3) BGR 图像, uint8
        Returns:
            list[dict]: 每个检测包含 {class_id, class_name, confidence, bbox(xyxy), is_person}
        """
        tensor, (ratio_w, ratio_h), (pad_w, pad_h) = self.preprocess(img_bgr)
        h0, w0 = img_bgr.shape[:2]

        outputs = self.session.run(None, {self.input_name: tensor})
        preds = outputs[0][0]   # (N, 6): [x1, y1, x2, y2, conf, cls]

        return self._postprocess(preds, (h0, w0), (ratio_w, ratio_h), (pad_w, pad_h))

    def _postprocess(self, preds, orig_shape, ratios, pads):
        """后处理: 置信度过滤 + 坐标逆变换 + clip."""
        ratio_w, ratio_h = ratios
        pad_w, pad_h = pads
        h0, w0 = orig_shape

        # 置信度过滤
        keep = preds[:, 4] > self.conf
        preds = preds[keep]

        if len(preds) == 0:
            return []

        # bbox 坐标: model space → original image space
        # 逆序: 先减去填充, 再乘以缩放因子
        boxes = preds[:, :4].copy()
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, [0, 2]] *= ratio_w
        boxes[:, [1, 3]] *= ratio_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        results = []
        for i in range(len(preds)):
            cls_id = int(preds[i, 5])
            if cls_id >= len(FINETUNE_CLASSES):
                continue
            results.append({
                "class_id": cls_id,
                "class_name": FINETUNE_CLASSES[cls_id],
                "confidence": float(preds[i, 4]),
                "bbox": boxes[i].astype(np.float32),    # (4,) xyxy
                "is_person": cls_id in (0, 1),
            })

        return results


# ==============================================================================
# YOLOSegONNX: 分割模型推理 (保留, 需要 mask 时可切换)
# ==============================================================================

class YOLOSegONNX:
    """YOLOv8 实例分割 ONNX 推理.

    输出格式 (两个 tensor):
      output0: (1, 116, 8400)   — 4 bbox + 80 cls + 32 mask_coeffs
      output1: (1, 32, 160, 160) — prototype masks

    后处理比 YOLODetectONNX 重得多:
      置信度过滤 → NMS → sigmoid(mask_coeffs @ proto) → threshold → per-instance mask
    """

    def __init__(self, onnx_path, conf_thresh=0.5, iou_thresh=0.65, imgsz=640):
        self.conf = conf_thresh
        self.iou = iou_thresh
        self.imgsz = imgsz

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]
        self.num_classes = 80
        self.num_masks = 32       # prototype mask 数量 (标准 YOLOv8-seg)

        print(f"[YOLOSegONNX] Loaded {onnx_path}")
        print(f"  Input: {self.input_name} {input_shape}")
        for o in self.session.get_outputs():
            print(f"  Output: {o.name} {o.shape}")

    def preprocess(self, img_bgr):
        """Letterbox 预处理, 与 YOLODetectONNX.preprocess 相同."""
        h0, w0 = img_bgr.shape[:2]
        r = min(self.input_w / w0, self.input_h / h0)
        new_w, new_h = int(w0 * r), int(h0 * r)
        ratio_w, ratio_h = w0 / new_w, h0 / new_h

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw = (self.input_w - new_w) / 2
        dh = (self.input_h - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

        tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, (ratio_w, ratio_h), (left, top)

    def predict(self, img_bgr):
        """运行分割推理, 返回 bbox + mask."""
        tensor, (ratio_w, ratio_h), (pad_w, pad_h) = self.preprocess(img_bgr)
        h0, w0 = img_bgr.shape[:2]

        outputs = self.session.run(None, {self.input_name: tensor})
        preds = outputs[0][0]               # (116, 8400)
        proto = outputs[1][0] if len(outputs) > 1 else None  # (32, 160, 160)

        return self._postprocess(preds, proto, (h0, w0),
                                 (ratio_w, ratio_h), (pad_w, pad_h))

    def _postprocess(self, preds, proto, orig_shape, ratios, pads):
        """完整后处理: 拆分输出 + 解码 + NMS + mask 生成."""
        preds = np.ascontiguousarray(preds.T)       # (8400, 116)
        ratio_w, ratio_h = ratios
        pad_w, pad_h = pads
        h0, w0 = orig_shape

        # 拆分: 前 4 列 bbox, 中间 80 列分类, 后 32 列 mask 系数
        bboxes_raw = preds[:, :4]
        cls_logits = preds[:, 4:4 + self.num_classes]
        mask_coeffs = preds[:, 4 + self.num_classes:]

        scores = cls_logits.max(axis=1)
        class_ids = cls_logits.argmax(axis=1)

        keep = scores > self.conf
        if not keep.any():
            return []

        bboxes_raw = bboxes_raw[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        boxes = self._cxcywh_to_xyxy(bboxes_raw)

        # model space → original image space
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes[:, [0, 2]] *= ratio_w
        boxes[:, [1, 3]] *= ratio_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        # NMS (分割模型导出时通常不内置 NMS, 需要手动做)
        keep_idx = self._nms(boxes, scores, self.iou)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]
        mask_coeffs = mask_coeffs[keep_idx]

        masks = self._generate_masks(proto, mask_coeffs, boxes, (h0, w0),
                                     (self.input_h, self.input_w), pads)

        results = []
        for i in range(len(boxes)):
            cls_id = int(class_ids[i])
            results.append({
                "class_id": cls_id,
                "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown",
                "confidence": float(scores[i]),
                "bbox": boxes[i].astype(np.float32),
                "mask": masks[i] if masks is not None else None,
                "is_person": cls_id == 0,
            })

        return results

    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        """cx,cy,w,h → x1,y1,x2,y2."""
        out = np.zeros_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    @staticmethod
    def _nms(boxes, scores, iou_thresh):
        """NumPy 实现的非极大值抑制 (NMS)."""
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_thresh]

        return np.array(keep, dtype=np.int64)

    def _generate_masks(self, proto, mask_coeffs, boxes, orig_shape, input_shape, pads):
        """mask_coeffs × prototype masks → sigmoid → threshold → 每实例 mask.

        流程:
          proto (32, 160, 160) 是全局的 mask 基,
          mask_coeffs (N, 32) 是每个实例的线性组合系数,
          矩阵乘法得到 N 个 (160, 160) 的 mask, sigmoid 后 resize 到原始尺寸.
        """
        if proto is None or len(mask_coeffs) == 0:
            return None

        proto_flat = proto.reshape(self.num_masks, -1)        # (32, 160×160)
        masks = mask_coeffs @ proto_flat                       # (N, 160×160)
        masks = masks.reshape(-1, 160, 160)                    # (N, 160, 160)
        masks = 1.0 / (1.0 + np.exp(-masks))                   # sigmoid: → [0,1]

        pad_w, pad_h = pads
        h0, w0 = orig_shape
        result = []
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)
            if x2 <= x1 or y2 <= y1:
                result.append(np.zeros((h0, w0), dtype=bool))
                continue
            # resize mask 到 bbox 尺寸, 放置到全图
            mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1),
                                       interpolation=cv2.INTER_LINEAR)
            full_mask = np.zeros((h0, w0), dtype=bool)
            full_mask[y1:y2, x1:x2] = mask_resized > 0.5
            result.append(full_mask)

        return result


# ==============================================================================
# ONNX 导出工具 (从 ultralytics 模型)
# ==============================================================================

def export_onnx(model_name="yolov8s-seg", imgsz=640, output_dir="models"):
    """从 ultralytics 导出 YOLO 模型到 ONNX.

    用法:
      export_onnx("yolov8s-seg", imgsz=640)  → models/yolov8s-seg.onnx
      export_onnx("yolov8n", imgsz=640)      → models/yolov8n.onnx
    """
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    if onnx_path.exists():
        print(f"[Export] {onnx_path} already exists, skipping")
        return onnx_path

    print(f"[Export] Downloading {model_name}.pt ...")
    model = YOLO(f"{model_name}.pt")
    print(f"[Export] Exporting to ONNX (imgsz={imgsz}, simplified)...")
    success = model.export(format="onnx", imgsz=imgsz, simplify=True)

    if isinstance(success, str):
        src = Path(success)
        if src != onnx_path:
            src.rename(onnx_path)
        print(f"[Export] Saved to {onnx_path}")

    return onnx_path


# ==============================================================================
# YOLOPtDetector: ultralytics .pt 模型包装器 (支持微调模型)
# ==============================================================================

class YOLOPtDetector:
    """用 ultralytics 直接加载 .pt 模型做推理.

    输出格式与 YOLODetectONNX 一致, 可直接替换.
    """

    def __init__(self, pt_path, conf_thresh=0.25, imgsz=640, device='cuda'):
        from ultralytics import YOLO
        self.model = YOLO(pt_path)
        self.conf = conf_thresh
        self.imgsz = imgsz
        self.device = device
        # 从模型读取类别名
        if hasattr(self.model.model, 'names'):
            self.class_names = self.model.model.names
        else:
            self.class_names = FINETUNE_CLASSES
        print(f"[YOLOPtDetector] Loaded {pt_path}")
        print(f"  Classes: {self.class_names}")

    def predict(self, img_bgr):
        """Returns: list[dict] 同 YOLODetectONNX.predict 格式."""
        h0, w0 = img_bgr.shape[:2]
        results = self.model(img_bgr, conf=self.conf, imgsz=self.imgsz,
                             device=self.device, verbose=False)

        dets = []
        if len(results) == 0:
            return dets

        r = results[0]
        if r.boxes is None:
            return dets

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes_xyxy)):
            cls_id = cls_ids[i]
            dets.append({
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, f"cls_{cls_id}"),
                "confidence": float(confs[i]),
                "bbox": boxes_xyxy[i].astype(np.float32),
                "is_person": cls_id in (0, 1),
            })

        return dets


if __name__ == "__main__":
    # 烟雾测试
    onnx_path = export_onnx("yolov8s-seg", imgsz=640, output_dir="models")
    detector = YOLOSegONNX(onnx_path, conf_thresh=0.3)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = detector.predict(dummy)
    print(f"Dummy inference: {len(dets)} detections")
