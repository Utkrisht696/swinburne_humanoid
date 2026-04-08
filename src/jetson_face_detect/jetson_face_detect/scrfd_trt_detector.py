#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import ctypes

import cv2
import numpy as np
import tensorrt as trt


@dataclass
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    keypoints: Optional[np.ndarray] = None  # shape (5, 2)


class CudaRt:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int

        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def check(self, err: int, what: str):
        if err != 0:
            raise RuntimeError(f"{what} failed with CUDA error code {err}")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr.value

    def free(self, ptr: int):
        if ptr:
            self.check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def memcpy_htod(self, dst_ptr: int, src_np: np.ndarray):
        src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst_ptr),
                src_ptr,
                src_np.nbytes,
                self.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy HtoD",
        )

    def memcpy_htod_async(self, dst_ptr: int, src_np: np.ndarray, stream: int):
        src_ptr = src_np.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                src_ptr,
                src_np.nbytes,
                self.cudaMemcpyHostToDevice,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync HtoD",
        )

    def memcpy_dtoh(self, dst_np: np.ndarray, src_ptr: int):
        dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(
                dst_ptr,
                ctypes.c_void_p(src_ptr),
                dst_np.nbytes,
                self.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy DtoH",
        )

    def memcpy_dtoh_async(self, dst_np: np.ndarray, src_ptr: int, stream: int):
        dst_ptr = dst_np.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpyAsync(
                dst_ptr,
                ctypes.c_void_p(src_ptr),
                dst_np.nbytes,
                self.cudaMemcpyDeviceToHost,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync DtoH",
        )

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream.value

    def stream_destroy(self, stream: int):
        if stream:
            self.check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")

    def stream_synchronize(self, stream: int):
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


class SCRFDTensorRTDetector:
    """
    SCRFD TensorRT detector for det_2.5g_fp16.engine.
    Assumes 640x640 input and 3 output strides: 8, 16, 32.
    Preprocessing/decode follows the official SCRFD style more closely:
      - keep aspect ratio
      - resize
      - paste at top-left on zero canvas
      - blobFromImage(..., mean=127.5, std=128, swapRB=True)
      - divide decoded boxes/keypoints by a single det_scale
    """

    def __init__(
        self,
        engine_path: str,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.4,
        max_num: int = 20,
        input_size: Tuple[int, int] = (640, 640),
        apply_sigmoid: bool = False,
        debug: bool = False,
    ):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_num = max_num
        self.input_width = input_size[0]
        self.input_height = input_size[1]
        self.apply_sigmoid = apply_sigmoid
        self.debug = debug

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.cuda = CudaRt()
        self.stream = self.cuda.stream_create()

        self.input_name = None
        self.output_names: List[str] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if self.input_name is None:
            raise RuntimeError("No input tensor found in engine")

        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if -1 in input_shape:
            self.context.set_input_shape(self.input_name, (1, 3, self.input_height, self.input_width))

        self.device_buffers: Dict[str, int] = {}
        self.host_outputs: Dict[str, np.ndarray] = {}

        input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        self.input_shape = input_shape
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.input_nbytes = int(np.prod(input_shape) * np.dtype(self.input_dtype).itemsize)
        self.device_buffers[self.input_name] = self.cuda.malloc(self.input_nbytes)
        self.context.set_tensor_address(self.input_name, int(self.device_buffers[self.input_name]))

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_arr = np.empty(shape, dtype=dtype)
            self.host_outputs[name] = host_arr
            self.device_buffers[name] = self.cuda.malloc(host_arr.nbytes)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        if self.debug:
            print("[SCRFD] input:", self.input_name, self.input_shape)
            for name in self.output_names:
                print("[SCRFD] output:", name, self.context.get_tensor_shape(name), self.engine.get_tensor_dtype(name))

        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.anchor_centers: Dict[int, np.ndarray] = {}
        for stride in self.strides:
            feat_h = self.input_height // stride
            feat_w = self.input_width // stride
            yy, xx = np.mgrid[:feat_h, :feat_w]
            centers = np.stack([xx, yy], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self.anchor_centers[stride] = centers

    def __del__(self):
        try:
            for ptr in self.device_buffers.values():
                self.cuda.free(ptr)
            self.cuda.stream_destroy(self.stream)
        except Exception:
            pass

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame_bgr.shape[:2]

        blob, det_scale = self.preprocess(frame_bgr)

        self.cuda.memcpy_htod_async(self.device_buffers[self.input_name], blob, self.stream)
        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        for name in self.output_names:
            host_arr = self.host_outputs[name]
            self.cuda.memcpy_dtoh_async(host_arr, self.device_buffers[name], self.stream)
        self.cuda.stream_synchronize(self.stream)

        if self.debug:
            for name, arr in self.host_outputs.items():
                a = arr.astype(np.float32, copy=False)
                print(f"[SCRFD] {name} shape={a.shape} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")

        dets = self.decode_outputs(self.host_outputs, orig_w, orig_h, det_scale)
        return dets

    def preprocess(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        target_w, target_h = self.input_width, self.input_height

        im_ratio = float(h) / float(w)
        model_ratio = float(target_h) / float(target_w)

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * im_ratio)

        det_scale = float(new_h) / float(h)

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            det_img,
            scalefactor=1.0 / 128.0,
            size=(target_w, target_h),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        ).astype(self.input_dtype, copy=False)

        return blob, det_scale

    def decode_outputs(
        self,
        outputs: Dict[str, np.ndarray],
        orig_w: int,
        orig_h: int,
        det_scale: float,
    ) -> List[Detection]:
        groups = self.group_outputs(outputs)

        all_scores = []
        all_boxes = []
        all_kps = []

        for stride, tensors in groups.items():
            scores = tensors["score"].reshape(-1).astype(np.float32, copy=False)
            if self.apply_sigmoid:
                scores = 1.0 / (1.0 + np.exp(-scores))

            boxes = tensors["bbox"].reshape(-1, 4).astype(np.float32, copy=False)
            kps = tensors["kps"].reshape(-1, 10).astype(np.float32, copy=False)

            if self.debug:
                print(f"[SCRFD] stride={stride} score min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")

            centers = self.anchor_centers[stride]
            if centers.shape[0] != scores.shape[0]:
                raise RuntimeError(
                    f"SCRFD output/anchor mismatch for stride {stride}: "
                    f"{centers.shape[0]} anchors vs {scores.shape[0]} scores"
                )

            keep = np.where(scores >= self.conf_threshold)[0]
            if self.debug:
                print(f"[SCRFD] stride={stride} keep={keep.size}")

            if keep.size == 0:
                continue

            scores = scores[keep]
            boxes = boxes[keep] * float(stride)
            kps = kps[keep] * float(stride)
            centers = centers[keep]

            x1 = centers[:, 0] - boxes[:, 0]
            y1 = centers[:, 1] - boxes[:, 1]
            x2 = centers[:, 0] + boxes[:, 2]
            y2 = centers[:, 1] + boxes[:, 3]
            det_boxes = np.stack([x1, y1, x2, y2], axis=1)

            det_kps = np.zeros((kps.shape[0], 5, 2), dtype=np.float32)
            for i in range(5):
                det_kps[:, i, 0] = centers[:, 0] + kps[:, 2 * i + 0]
                det_kps[:, i, 1] = centers[:, 1] + kps[:, 2 * i + 1]

            all_scores.append(scores)
            all_boxes.append(det_boxes)
            all_kps.append(det_kps)

        if not all_scores:
            return []

        scores = np.concatenate(all_scores, axis=0)
        boxes = np.concatenate(all_boxes, axis=0)
        kps = np.concatenate(all_kps, axis=0)

        keep = self.nms(boxes, scores, self.nms_threshold)
        if self.max_num > 0:
            keep = keep[: self.max_num]

        detections: List[Detection] = []
        for idx in keep:
            box = boxes[idx].copy()
            pts = kps[idx].copy()

            box /= det_scale
            pts /= det_scale

            x1 = int(np.clip(box[0], 0, orig_w - 1))
            y1 = int(np.clip(box[1], 0, orig_h - 1))
            x2 = int(np.clip(box[2], 0, orig_w - 1))
            y2 = int(np.clip(box[3], 0, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)

            detections.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    score=float(scores[idx]),
                    keypoints=pts.astype(np.float32),
                )
            )

        return detections

    def group_outputs(self, outputs: Dict[str, np.ndarray]):
        buckets: Dict[int, Dict[str, np.ndarray]] = {}

        for name, arr in outputs.items():
            a = np.squeeze(arr)
            if a.ndim == 1:
                a = a[:, None]

            rows, cols = a.shape
            if self.debug:
                print(f"[SCRFD] group rows={rows}, cols={cols}, name={name}")

            if rows not in buckets:
                buckets[rows] = {}

            if cols == 1:
                buckets[rows]["score"] = a
            elif cols == 4:
                buckets[rows]["bbox"] = a
            elif cols == 10:
                buckets[rows]["kps"] = a
            else:
                raise RuntimeError(f"Unexpected SCRFD output tensor shape for {name}: {a.shape}")

        grouped: Dict[int, Dict[str, np.ndarray]] = {}
        row_to_stride = {12800: 8, 3200: 16, 800: 32}

        for rows, tensors in buckets.items():
            if rows not in row_to_stride:
                raise RuntimeError(f"Unknown SCRFD row count {rows}, cannot map to stride")
            stride = row_to_stride[rows]
            if not all(k in tensors for k in ("score", "bbox", "kps")):
                raise RuntimeError(f"Incomplete SCRFD tensor group for stride {stride}")
            grouped[stride] = tensors

        return grouped

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            rest = order[1:]

            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            union = area_i + area_r - inter
            iou = np.where(union > 0, inter / union, 0.0)

            inds = np.where(iou <= iou_thresh)[0]
            order = rest[inds]
        return keep
