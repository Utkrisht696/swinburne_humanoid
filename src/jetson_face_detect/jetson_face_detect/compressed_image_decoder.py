#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import gi

    gi.require_version('Gst', '1.0')
    gi.require_version('GstApp', '1.0')
    from gi.repository import Gst

    Gst.init(None)
    GST_AVAILABLE = True
except Exception:
    Gst = None
    GST_AVAILABLE = False


@dataclass
class DecodeResult:
    frame_bgr: Optional[np.ndarray]
    backend: str


class CompressedImageDecoder:
    def __init__(self, prefer_hw: bool = True):
        self.prefer_hw = prefer_hw
        self.backend_status = 'opencv-imdecode'
        self.pipeline = None
        self.appsrc = None
        self.appsink = None
        self.output_bgr: Optional[np.ndarray] = None
        self.sample_timeout_ns = int(0.2 * 1e9)
        self._gst_pts = 0
        if self.prefer_hw:
            self._init_gstreamer_decoder()

    def _init_gstreamer_decoder(self):
        if not GST_AVAILABLE:
            self.backend_status = 'opencv-imdecode (GStreamer unavailable)'
            return

        pipeline_desc = (
            'appsrc name=src is-live=false block=true format=time caps=image/jpeg '
            '! nvjpegdec mjpegdecode=true '
            '! video/x-raw,format=I420 '
            '! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true wait-on-eos=false'
        )

        try:
            pipeline = Gst.parse_launch(pipeline_desc)
            appsrc = pipeline.get_by_name('src')
            appsink = pipeline.get_by_name('sink')
            if pipeline is None or appsrc is None or appsink is None:
                self.backend_status = 'opencv-imdecode (failed to build GStreamer JPEG pipeline)'
                return
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                self.backend_status = 'opencv-imdecode (failed to start nvjpegdec pipeline)'
                return
            self.pipeline = pipeline
            self.appsrc = appsrc
            self.appsink = appsink
            self.backend_status = 'gstreamer-nvjpegdec-i420'
        except Exception as exc:
            self.backend_status = f'opencv-imdecode (nvjpegdec unavailable: {exc})'

    def decode(self, jpeg_bytes: bytes) -> DecodeResult:
        if self.pipeline is not None:
            frame = self._decode_with_gstreamer(jpeg_bytes)
            if frame is not None:
                return DecodeResult(frame_bgr=frame, backend=self.backend_status)
        return DecodeResult(frame_bgr=self._decode_with_opencv(jpeg_bytes), backend='opencv-imdecode')

    def _decode_with_gstreamer(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        if self.appsrc is None or self.appsink is None:
            return None

        try:
            gst_buffer = Gst.Buffer.new_allocate(None, len(jpeg_bytes), None)
            gst_buffer.fill(0, jpeg_bytes)
            gst_buffer.pts = self._gst_pts
            gst_buffer.dts = self._gst_pts
            self._gst_pts += int(1e9 / 30.0)
            flow_ret = self.appsrc.emit('push-buffer', gst_buffer)
            if flow_ret != Gst.FlowReturn.OK:
                return None

            sample = self.appsink.emit('try-pull-sample', self.sample_timeout_ns)
            if sample is None:
                return None

            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            buffer = sample.get_buffer()
            ok, map_info = buffer.map(Gst.MapFlags.READ)
            if not ok:
                return None

            try:
                i420 = np.ndarray((height * 3 // 2, width), buffer=map_info.data, dtype=np.uint8)
                if self.output_bgr is None or self.output_bgr.shape[:2] != (height, width):
                    self.output_bgr = np.empty((height, width, 3), dtype=np.uint8)
                cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420, dst=self.output_bgr)
                return self.output_bgr
            finally:
                buffer.unmap(map_info)
        except Exception:
            return None

    def _decode_with_opencv(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def close(self):
        if self.appsrc is not None:
            try:
                self.appsrc.emit('end-of-stream')
            except Exception:
                pass
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self.pipeline = None
        self.appsrc = None
        self.appsink = None

    def get_backend_status(self) -> str:
        return self.backend_status
