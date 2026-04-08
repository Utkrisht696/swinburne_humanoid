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


@dataclass
class HostBuffer:
    ptr: int
    array: np.ndarray
    nbytes: int
    pinned: bool = False


class CudaRt:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaHostAllocDefault = 0

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
        self.lib.cudaHostAlloc.restype = ctypes.c_int

        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaFreeHost.restype = ctypes.c_int

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

    def host_alloc(self, shape: Tuple[int, ...], dtype: np.dtype) -> HostBuffer:
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape) * dtype.itemsize)
        ptr = ctypes.c_void_p()
        err = self.lib.cudaHostAlloc(ctypes.byref(ptr), nbytes, self.cudaHostAllocDefault)
        if err != 0:
            array = np.empty(shape, dtype=dtype)
            return HostBuffer(ptr=0, array=array, nbytes=array.nbytes, pinned=False)
        flat = np.ctypeslib.as_array((ctypes.c_byte * nbytes).from_address(ptr.value))
        array = flat.view(dtype=dtype).reshape(shape)
        return HostBuffer(ptr=ptr.value, array=array, nbytes=nbytes, pinned=True)

    def free_host(self, ptr: int):
        if ptr:
            self.check(self.lib.cudaFreeHost(ctypes.c_void_p(ptr)), "cudaFreeHost")

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
    Preprocessing keeps aspect ratio and writes into persistent host/device buffers.
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
        use_cuda_preprocess: bool = True,
    ):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_num = max_num
        self.input_width = input_size[0]
        self.input_height = input_size[1]
        self.apply_sigmoid = apply_sigmoid
        self.debug = debug
        self.use_cuda_preprocess = use_cuda_preprocess

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
        self.host_output_buffers: Dict[str, HostBuffer] = {}

        input_shape = tuple(self.context.get_tensor_shape(self.input_name))
        self.input_shape = input_shape
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.input_nbytes = int(np.prod(input_shape) * np.dtype(self.input_dtype).itemsize)
        self.host_input = self.cuda.host_alloc(self.input_shape, self.input_dtype)
        self.device_buffers[self.input_name] = self.cuda.malloc(self.input_nbytes)
        self.context.set_tensor_address(self.input_name, int(self.device_buffers[self.input_name]))

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buf = self.cuda.host_alloc(shape, dtype)
            self.host_output_buffers[name] = host_buf
            self.host_outputs[name] = host_buf.array
            self.device_buffers[name] = self.cuda.malloc(host_buf.nbytes)
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        if self.debug:
            print("[SCRFD] input:", self.input_name, self.input_shape)
            for name in self.output_names:
                print("[SCRFD] output:", name, self.context.get_tensor_shape(name), self.engine.get_tensor_dtype(name))

        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.det_img = np.empty((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.det_img.fill(0)
        self.det_img_rgb = np.empty((self.input_height, self.input_width, 3), dtype=np.uint8)
        self._channel_pad_value = np.array([(0.0 - 127.5) / 128.0], dtype=self.input_dtype)[0]
        self._last_resize_shape: Optional[Tuple[int, int]] = None
        self.preprocess_backend_status = "cpu (forced by parameter)"
        self.cuda_preprocess_enabled = self._init_cuda_preprocess()
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
            self.cuda.free_host(self.host_input.ptr)
            for host_buf in self.host_output_buffers.values():
                self.cuda.free_host(host_buf.ptr)
            self.cuda.stream_destroy(self.stream)
        except Exception:
            pass

    def _init_cuda_preprocess(self) -> bool:
        try:
            if not self.use_cuda_preprocess:
                self.preprocess_backend_status = "cpu (forced by parameter)"
                return False
            if not hasattr(cv2, "cuda"):
                self.preprocess_backend_status = "cpu (OpenCV CUDA module unavailable)"
                return False
            if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
                self.preprocess_backend_status = "cpu (no CUDA device available to OpenCV)"
                return False
            self.gpu_src = cv2.cuda_GpuMat()
            self.gpu_resized_bgr = cv2.cuda_GpuMat()
            self.gpu_canvas_bgr = cv2.cuda_GpuMat()
            self.gpu_canvas_rgb = cv2.cuda_GpuMat()
            self.gpu_canvas_bgr.create(self.input_height, self.input_width, cv2.CV_8UC3)
            self.gpu_canvas_rgb.create(self.input_height, self.input_width, cv2.CV_8UC3)
            self.preprocess_backend_status = "cuda"
            if self.debug:
                print("[SCRFD] CUDA preprocess enabled")
            return True
        except Exception as exc:
            self.preprocess_backend_status = f"cpu (CUDA init failed: {exc})"
            if self.debug:
                print(f"[SCRFD] CUDA preprocess disabled: {exc}")
            return False

    def detect(self, frame_bgr: np.ndarray, timings: Optional[Dict[str, float]] = None) -> List[Detection]:
        orig_h, orig_w = frame_bgr.shape[:2]
        if timings is None:
            timings = {}

        t0 = cv2.getTickCount()
        det_scale = self.preprocess(frame_bgr, timings)
        t1 = cv2.getTickCount()

        self.cuda.memcpy_htod_async(self.device_buffers[self.input_name], self.host_input.array, self.stream)
        t2 = cv2.getTickCount()
        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        t3 = cv2.getTickCount()
        for name in self.output_names:
            host_arr = self.host_outputs[name]
            self.cuda.memcpy_dtoh_async(host_arr, self.device_buffers[name], self.stream)
        self.cuda.stream_synchronize(self.stream)
        t4 = cv2.getTickCount()

        if self.debug:
            for name, arr in self.host_outputs.items():
                a = arr.astype(np.float32, copy=False)
                print(f"[SCRFD] {name} shape={a.shape} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")

        tick_ms = 1000.0 / cv2.getTickFrequency()
        timings["preprocess_ms"] = (t1 - t0) * tick_ms
        timings["h2d_ms"] = (t2 - t1) * tick_ms
        timings["infer_enqueue_ms"] = (t3 - t2) * tick_ms
        timings["dtoh_sync_ms"] = (t4 - t3) * tick_ms

        dets = self.decode_outputs(self.host_outputs, orig_w, orig_h, det_scale)
        return dets

    def preprocess(self, image_bgr: np.ndarray, timings: Optional[Dict[str, float]] = None) -> float:
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
        if timings is None:
            timings = {}
        cuda_start = cv2.getTickCount()
        source_rgb = self._preprocess_resize_and_color(image_bgr, new_w, new_h)
        cuda_end = cv2.getTickCount()
        timings["resize_color_ms"] = (cuda_end - cuda_start) * (1000.0 / cv2.getTickFrequency())

        pack_start = cv2.getTickCount()
        blob = self.host_input.array[0]
        if self._last_resize_shape != (new_h, new_w):
            blob.fill(self._channel_pad_value)
            self._last_resize_shape = (new_h, new_w)
        else:
            if new_h < target_h:
                blob[:, new_h:, :] = self._channel_pad_value
            if new_w < target_w:
                blob[:, :new_h, new_w:] = self._channel_pad_value

        active = source_rgb[:new_h, :new_w]
        np.copyto(blob[0, :new_h, :new_w], active[:, :, 2], casting="unsafe")
        np.copyto(blob[1, :new_h, :new_w], active[:, :, 1], casting="unsafe")
        np.copyto(blob[2, :new_h, :new_w], active[:, :, 0], casting="unsafe")
        blob[:, :new_h, :new_w] -= 127.5
        blob[:, :new_h, :new_w] *= (1.0 / 128.0)
        pack_end = cv2.getTickCount()
        timings["tensor_pack_ms"] = (pack_end - pack_start) * (1000.0 / cv2.getTickFrequency())
        timings["preprocess_ms"] = timings["resize_color_ms"] + timings["tensor_pack_ms"]

        return det_scale

    def _preprocess_resize_and_color(self, image_bgr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        if self.cuda_preprocess_enabled:
            try:
                self.gpu_src.upload(image_bgr)
                self.gpu_resized_bgr = cv2.cuda.resize(
                    self.gpu_src,
                    (new_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                self.gpu_canvas_bgr.setTo((0, 0, 0))
                self.gpu_resized_bgr.copyTo(self.gpu_canvas_bgr.rowRange(0, new_h).colRange(0, new_w))
                self.gpu_canvas_rgb = cv2.cuda.cvtColor(self.gpu_canvas_bgr, cv2.COLOR_BGR2RGB)
                self.gpu_canvas_rgb.download(self.det_img_rgb)
                return self.det_img_rgb
            except Exception as exc:
                self.cuda_preprocess_enabled = False
                self.preprocess_backend_status = f"cpu (CUDA runtime fallback: {exc})"
                if self.debug:
                    print(f"[SCRFD] CUDA preprocess failed, falling back to CPU: {exc}")

        scale_x = float(new_w) / float(image_bgr.shape[1])
        scale_y = float(new_h) / float(image_bgr.shape[0])
        warp = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
            ],
            dtype=np.float32,
        )
        cv2.warpAffine(
            image_bgr,
            warp,
            (self.input_width, self.input_height),
            dst=self.det_img,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cv2.cvtColor(self.det_img, cv2.COLOR_BGR2RGB, dst=self.det_img_rgb)
        return self.det_img_rgb

    def get_preprocess_backend_status(self) -> str:
        return self.preprocess_backend_status

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
