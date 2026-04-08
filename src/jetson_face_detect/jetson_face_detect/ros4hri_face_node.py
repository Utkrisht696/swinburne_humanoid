#!/usr/bin/env python3

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import tf_transformations
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from tf2_ros import TransformBroadcaster

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from hri_msgs.msg import IdsList, NormalizedRegionOfInterest2D, FacialLandmarks

from .scrfd_trt_detector import SCRFDTensorRTDetector
from .face_tracker import SimpleFaceTracker


class Ros4HriFaceNode(Node):
    def __init__(self):
        super().__init__('ros4hri_face_node')

        self.declare_parameter(
            'image_topic',
            '/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed'
        )
        self.declare_parameter('camera_info_topic', '/aima/hal/sensor/rgb_head_front_center/camera_info')
        self.declare_parameter(
            'camera_matrix',
            [
                1449.2708332795, 0.0, 1329.5378435058,
                0.0, 1449.1918598406, 878.8810189435,
                0.0, 0.0, 1.0,
            ],
        )
        self.declare_parameter(
            'dist_coeffs',
            [
                28.2055207357,
                14.2650504336,
                -0.0001410914,
                9.30159e-05,
                0.6060234194,
                28.6276312274,
                25.7199699164,
                3.7955422286,
            ],
        )
        self.declare_parameter('frame_id', 'rgb_head_center')
        self.declare_parameter('engine_path', '/home/agi/models/det_2.5g_fp16.engine')
        self.declare_parameter('debug_width', 960)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('publish_landmarks', True)
        self.declare_parameter('crop_size', 128)
        self.declare_parameter('max_faces', 10)
        self.declare_parameter('conf_threshold', 0.10)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('min_face_size', 20)
        self.declare_parameter('publish_min_width', 40)
        self.declare_parameter('publish_min_height', 40)
        self.declare_parameter('publish_min_area', 2500)
        self.declare_parameter('publish_min_aspect_ratio', 0.45)
        self.declare_parameter('publish_max_aspect_ratio', 1.8)
        self.declare_parameter('tracker_iou_threshold', 0.35)
        self.declare_parameter('tracker_min_predicted_iou', 0.05)
        self.declare_parameter('tracker_max_center_distance_ratio', 0.9)
        self.declare_parameter('tracker_max_missed', 6)
        self.declare_parameter('tracker_debug', False)
        self.declare_parameter('performance_debug', True)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_crops', True)
        self.declare_parameter('publish_aligned', True)
        self.declare_parameter('apply_sigmoid', True)
        self.declare_parameter('detector_debug', True)
        self.declare_parameter('tf_position_smoothing', 0.25)
        self.declare_parameter('tf_rotation_smoothing', 0.2)

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_matrix_param = list(self.get_parameter('camera_matrix').value)
        self.dist_coeffs_param = list(self.get_parameter('dist_coeffs').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.engine_path = self.get_parameter('engine_path').value
        self.debug_width = int(self.get_parameter('debug_width').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.publish_landmarks = bool(self.get_parameter('publish_landmarks').value)
        self.crop_size = int(self.get_parameter('crop_size').value)
        self.max_faces = int(self.get_parameter('max_faces').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.nms_threshold = float(self.get_parameter('nms_threshold').value)
        self.min_face_size = int(self.get_parameter('min_face_size').value)
        self.publish_min_width = int(self.get_parameter('publish_min_width').value)
        self.publish_min_height = int(self.get_parameter('publish_min_height').value)
        self.publish_min_area = int(self.get_parameter('publish_min_area').value)
        self.publish_min_aspect_ratio = float(self.get_parameter('publish_min_aspect_ratio').value)
        self.publish_max_aspect_ratio = float(self.get_parameter('publish_max_aspect_ratio').value)
        self.tracker_iou_threshold = float(self.get_parameter('tracker_iou_threshold').value)
        self.tracker_min_predicted_iou = float(self.get_parameter('tracker_min_predicted_iou').value)
        self.tracker_max_center_distance_ratio = float(self.get_parameter('tracker_max_center_distance_ratio').value)
        self.tracker_max_missed = int(self.get_parameter('tracker_max_missed').value)
        self.tracker_debug = bool(self.get_parameter('tracker_debug').value)
        self.performance_debug = bool(self.get_parameter('performance_debug').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.publish_crops = bool(self.get_parameter('publish_crops').value)
        self.publish_aligned = bool(self.get_parameter('publish_aligned').value)
        self.apply_sigmoid = bool(self.get_parameter('apply_sigmoid').value)
        self.detector_debug = bool(self.get_parameter('detector_debug').value)
        self.tf_position_smoothing = float(self.get_parameter('tf_position_smoothing').value)
        self.tf_rotation_smoothing = float(self.get_parameter('tf_rotation_smoothing').value)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_optical_frame = self.frame_id
        self.face_model_points = np.array(
            [
                [-30.0, 35.0, 30.0],   # left eye
                [30.0, 35.0, 30.0],    # right eye
                [0.0, 0.0, 0.0],       # nose tip
                [-25.0, -35.0, 20.0],  # left mouth
                [25.0, -35.0, 20.0],   # right mouth
            ],
            dtype=np.float32,
        )
        self.face_model_scale_m = 0.001
        self.face_width_m = 0.16
        self.face_rotation_correction = tf_transformations.quaternion_from_euler(np.pi, 0.0, 0.0)
        # Match the ROS4HRI reference face detector's gaze frame convention.
        # hri_engagement expects the gaze frame to use an optical-frame layout
        # where +Z points along the person's viewing direction.
        self.gaze_frame_rotation = tf_transformations.quaternion_from_euler(
            0.0,
            0.0,
            0.0,
        )
        self.warned_missing_camera_info = False
        self.warned_pose_estimation_failed = False
        self.received_camera_info = False
        self.face_tf_state: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.load_static_camera_info()

        self.detector = SCRFDTensorRTDetector(
            engine_path=self.engine_path,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            max_num=self.max_faces,
            input_size=(640, 640),
            apply_sigmoid=self.apply_sigmoid,
            debug=self.detector_debug,
        )
        self.tracker = SimpleFaceTracker(
            iou_threshold=self.tracker_iou_threshold,
            min_predicted_iou=self.tracker_min_predicted_iou,
            max_center_distance_ratio=self.tracker_max_center_distance_ratio,
            max_missed=self.tracker_max_missed,
            min_hits=1,
            debug=self.tracker_debug,
        )

        self.ros_face_id_map: Dict[int, str] = {}

        self.sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_cb,
            10
        )
        self.camera_info_sub_sensor = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.camera_info_sub_reliable = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_cb,
            reliable_qos,
        )

        self.pub_tracked = self.create_publisher(
            IdsList,
            '/humans/faces/tracked',
            10
        )
        self.pub_debug = self.create_publisher(
            CompressedImage,
            '/jetson_face_detect/debug_image/compressed',
            10
        )

        self.per_face_roi_pubs: Dict[str, any] = {}
        self.per_face_crop_pubs: Dict[str, any] = {}
        self.per_face_aligned_pubs: Dict[str, any] = {}
        self.per_face_landmarks_pubs: Dict[str, any] = {}

        self.frame_count = 0
        self.last_log_time = time.time()
        self.stage_totals = {
            'decode_ms': 0.0,
            'detect_ms': 0.0,
            'track_ms': 0.0,
            'publish_ms': 0.0,
            'debug_ms': 0.0,
        }

        self.get_logger().info(f'Subscribed to {self.image_topic}')
        self.get_logger().info(f'Subscribed to {self.camera_info_topic}')
        self.get_logger().info(f'Using SCRFD engine: {self.engine_path}')

    def load_static_camera_info(self):
        if len(self.camera_matrix_param) == 9:
            self.camera_matrix = np.array(self.camera_matrix_param, dtype=np.float64).reshape(3, 3)
        if self.dist_coeffs_param:
            self.dist_coeffs = np.array(self.dist_coeffs_param, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        if self.camera_matrix is not None:
            self.warned_missing_camera_info = False
            self.get_logger().info(
                f'Using static camera calibration for frame_id={self.camera_optical_frame}'
            )

    def camera_info_cb(self, msg: CameraInfo):
        if len(msg.k) == 9:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.d:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        if msg.header.frame_id:
            self.camera_optical_frame = msg.header.frame_id
        self.warned_missing_camera_info = False
        if not self.received_camera_info:
            self.get_logger().info(
                f'Received camera_info from {self.camera_info_topic} with frame_id={self.camera_optical_frame}'
            )
            self.received_camera_info = True

    def image_cb(self, msg: CompressedImage):
        t0 = time.perf_counter()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode compressed frame')
            return
        t1 = time.perf_counter()
        self.process_frame(msg.header, frame, decode_ms=(t1 - t0) * 1000.0)

    def process_frame(self, header, frame: np.ndarray, decode_ms: float):
        t_start_detect = time.perf_counter()
        should_publish_debug = (
            self.publish_debug_image and self.pub_debug.get_subscription_count() > 0
        )
        debug = None

        detections = self.detector.detect(frame)
        t2 = time.perf_counter()

        if debug is not None:
            for det in detections[:20]:
                x1, y1, x2, y2 = det.bbox_xyxy
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    debug,
                    f"{det.score:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        h, w = frame.shape[:2]
        tracks = self.tracker.update(detections)
        tracks = tracks[:self.max_faces]
        t3 = time.perf_counter()

        if self.tracker_debug and self.tracker.last_debug_events:
            self.log_tracker_events()

        ros_face_ids = []
        valid_tracks = 0
        active_face_tf_ids = set()
        active_track_ids = set()

        for trk in tracks:
            active_track_ids.add(trk.track_id)
            x1, y1, x2, y2 = self.clamp_box(trk.bbox_xyxy, w, h)
            bw = x2 - x1
            bh = y2 - y1
            aspect = bw / float(max(bh, 1))
            area = bw * bh

            if bw < max(self.min_face_size, self.publish_min_width):
                continue
            if bh < max(self.min_face_size, self.publish_min_height):
                continue
            if aspect < self.publish_min_aspect_ratio or aspect > self.publish_max_aspect_ratio:
                continue
            if area < self.publish_min_area:
                continue

            ros_face_id = self.get_ros_face_id(trk.track_id)
            ros_face_ids.append(ros_face_id)

            crop_box = self.expand_box(x1, y1, x2, y2, w, h)
            cx1, cy1, cx2, cy2 = crop_box
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            self.ensure_face_publishers(ros_face_id)
            self.publish_roi(header, ros_face_id, x1, y1, x2, y2, w, h)

            if self.publish_crops:
                crop_pub = self.per_face_crop_pubs[ros_face_id]
                if crop_pub.get_subscription_count() > 0:
                    crop_img = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                    self.publish_image(header, crop_pub, crop_img)

            if self.publish_aligned:
                aligned_pub = self.per_face_aligned_pubs[ros_face_id]
                if aligned_pub.get_subscription_count() > 0:
                    if trk.keypoints is not None and trk.keypoints.shape == (5, 2):
                        aligned_img = self.align_face(frame, trk.keypoints, self.crop_size)
                    else:
                        aligned_img = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                    self.publish_image(header, aligned_pub, aligned_img)

            if self.publish_landmarks:
                landmarks_pub = self.per_face_landmarks_pubs[ros_face_id]
                if landmarks_pub.get_subscription_count() > 0:
                    self.publish_landmarks_msg(
                        header, ros_face_id, x1, y1, x2, y2, w, h, trk.keypoints, frame
                    )

            self.publish_face_transforms(header, ros_face_id, (x1, y1, x2, y2), trk.keypoints)
            active_face_tf_ids.add(ros_face_id)

            if debug is not None:
                self.draw_tracked_debug(debug, ros_face_id, trk.score, (x1, y1, x2, y2), trk.keypoints)
            valid_tracks += 1

        stale_face_ids = [
            face_id for face_id in self.face_tf_state.keys()
            if face_id not in active_face_tf_ids
        ]
        for face_id in stale_face_ids:
            del self.face_tf_state[face_id]

        self.cleanup_stale_faces(active_track_ids)

        t4 = time.perf_counter()
        self.publish_tracked_ids(header, ros_face_ids)
        if should_publish_debug:
            if debug is None:
                debug = frame.copy()
                for det in detections[:20]:
                    x1, y1, x2, y2 = det.bbox_xyxy
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        debug,
                        f"{det.score:.2f}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                for trk in tracks:
                    x1, y1, x2, y2 = self.clamp_box(trk.bbox_xyxy, w, h)
                    bw = x2 - x1
                    bh = y2 - y1
                    aspect = bw / float(max(bh, 1))
                    area = bw * bh
                    if bw < max(self.min_face_size, self.publish_min_width):
                        continue
                    if bh < max(self.min_face_size, self.publish_min_height):
                        continue
                    if aspect < self.publish_min_aspect_ratio or aspect > self.publish_max_aspect_ratio:
                        continue
                    if area < self.publish_min_area:
                        continue
                    ros_face_id = self.get_ros_face_id(trk.track_id)
                    self.draw_tracked_debug(debug, ros_face_id, trk.score, (x1, y1, x2, y2), trk.keypoints)
            self.publish_debug(header, debug)
        t5 = time.perf_counter()
        self.log_fps(
            valid_tracks,
            len(detections),
            decode_ms=decode_ms,
            detect_ms=(t2 - t_start_detect) * 1000.0,
            track_ms=(t3 - t2) * 1000.0,
            publish_ms=(t4 - t3) * 1000.0,
            debug_ms=(t5 - t4) * 1000.0,
        )

    def get_ros_face_id(self, track_id: int) -> str:
        if track_id not in self.ros_face_id_map:
            self.ros_face_id_map[track_id] = f'jetson{track_id:02d}'
        return self.ros_face_id_map[track_id]

    def ensure_face_publishers(self, face_id: str):
        if face_id not in self.per_face_roi_pubs:
            self.per_face_roi_pubs[face_id] = self.create_publisher(
                NormalizedRegionOfInterest2D,
                f'/humans/faces/{face_id}/roi',
                10
            )
            self.per_face_crop_pubs[face_id] = self.create_publisher(
                Image,
                f'/humans/faces/{face_id}/cropped',
                10
            )
            self.per_face_aligned_pubs[face_id] = self.create_publisher(
                Image,
                f'/humans/faces/{face_id}/aligned',
                10
            )
            self.per_face_landmarks_pubs[face_id] = self.create_publisher(
                FacialLandmarks,
                f'/humans/faces/{face_id}/landmarks',
                10
            )

    def cleanup_stale_faces(self, active_track_ids):
        stale_track_ids = [
            track_id for track_id in self.ros_face_id_map.keys()
            if track_id not in active_track_ids
        ]
        for track_id in stale_track_ids:
            face_id = self.ros_face_id_map.pop(track_id, None)
            if face_id is None:
                continue
            for publisher_map in (
                self.per_face_roi_pubs,
                self.per_face_crop_pubs,
                self.per_face_aligned_pubs,
                self.per_face_landmarks_pubs,
            ):
                publisher = publisher_map.pop(face_id, None)
                if publisher is not None:
                    self.destroy_publisher(publisher)

    def clamp_box(self, bbox_xyxy: Tuple[int, int, int, int], w: int, h: int):
        x1, y1, x2, y2 = bbox_xyxy
        return (
            max(0, min(w - 1, int(x1))),
            max(0, min(h - 1, int(y1))),
            max(0, min(w - 1, int(x2))),
            max(0, min(h - 1, int(y2))),
        )

    def expand_box(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int):
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(0.18 * bw)
        pad_y_top = int(0.22 * bh)
        pad_y_bottom = int(0.12 * bh)

        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y_top)
        nx2 = min(w - 1, x2 + pad_x)
        ny2 = min(h - 1, y2 + pad_y_bottom)
        return nx1, ny1, nx2, ny2

    def draw_tracked_debug(self, image: np.ndarray, face_id: str, score: float, box, keypoints: Optional[np.ndarray]):
        x1, y1, x2, y2 = box
        # TRACKED/PUBLISHED box in green
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f'{face_id} {score:.2f}',
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        debug_points = self.compute_debug_landmark_points(x1, y1, x2, y2, keypoints)
        for px, py, color in debug_points:
            cv2.circle(image, (int(px), int(py)), 2, color, -1)

    def publish_tracked_ids(self, header, face_ids: List[str]):
        out = IdsList()
        out.header = header
        out.header.frame_id = self.frame_id
        out.ids = face_ids
        self.pub_tracked.publish(out)

    def publish_roi(self, header, face_id, x1, y1, x2, y2, w, h):
        roi = NormalizedRegionOfInterest2D()
        roi.header = header
        roi.header.frame_id = self.frame_id
        roi.xmin = float(x1) / float(w)
        roi.ymin = float(y1) / float(h)
        roi.xmax = float(x2) / float(w)
        roi.ymax = float(y2) / float(h)
        roi.c = 1.0
        self.per_face_roi_pubs[face_id].publish(roi)

    def publish_image(self, header, publisher, image_bgr: np.ndarray):
        out = self.bridge.cv2_to_imgmsg(image_bgr, encoding='bgr8')
        out.header = header
        out.header.frame_id = self.frame_id
        publisher.publish(out)

    def estimate_mouth_opening(
        self,
        frame: np.ndarray,
        mouth_left: np.ndarray,
        mouth_right: np.ndarray,
        mouth_center: np.ndarray,
        fallback_height: float,
    ) -> float:
        mouth_width = max(1.0, float(np.linalg.norm(mouth_right - mouth_left)))
        roi_half_w = int(max(6.0, 0.30 * mouth_width))
        roi_half_h = int(max(6.0, fallback_height))

        cx = int(round(float(mouth_center[0])))
        cy = int(round(float(mouth_center[1])))
        fh, fw = frame.shape[:2]
        rx1 = max(0, cx - roi_half_w)
        ry1 = max(0, cy - roi_half_h)
        rx2 = min(fw, cx + roi_half_w)
        ry2 = min(fh, cy + roi_half_h)

        if rx2 <= rx1 or ry2 <= ry1:
            return fallback_height

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return fallback_height

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        darkness = float(np.mean(255.0 - blur)) / 255.0

        row_profile = np.mean(255.0 - blur, axis=1) / 255.0
        active_rows = np.where(row_profile > max(0.10, 0.70 * np.max(row_profile)))[0]
        if active_rows.size >= 2:
            dark_span = float(active_rows[-1] - active_rows[0] + 1)
        else:
            dark_span = 0.0

        openness_from_span = 0.55 * dark_span
        openness_from_darkness = 0.22 * mouth_width * darkness
        estimated_height = max(fallback_height, openness_from_span + openness_from_darkness)
        return float(np.clip(estimated_height, 0.08 * mouth_width, 0.75 * mouth_width))

    def publish_landmarks_msg(
        self,
        header,
        face_id,
        x1,
        y1,
        x2,
        y2,
        w,
        h,
        keypoints: Optional[np.ndarray],
        frame: np.ndarray,
    ):
        lm = FacialLandmarks()
        lm.header = header
        lm.header.frame_id = self.frame_id
        lm.height = h
        lm.width = w

        for i in range(70):
            lm.landmarks[i].x = 0.0
            lm.landmarks[i].y = 0.0
            lm.landmarks[i].c = 0.0

        def put(idx: int, px: float, py: float, conf: float = 1.0):
            lm.landmarks[idx].x = max(0.0, min(1.0, px / w))
            lm.landmarks[idx].y = max(0.0, min(1.0, py / h))
            lm.landmarks[idx].c = conf

        def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            return (1.0 - t) * a + t * b

        def put_mouth(
            mouth_left: np.ndarray,
            mouth_right: np.ndarray,
            mouth_center: np.ndarray,
            mouth_height: float,
            conf: float = 1.0,
        ):
            width_vec = mouth_left - mouth_right
            mouth_width = max(1.0, float(np.linalg.norm(width_vec)))
            height_scale = max(2.0, float(mouth_height))
            lip_roundness = np.clip(height_scale / mouth_width, 0.08, 0.55)
            inner_height = max(1.0, 0.55 * height_scale)

            def lip_arc(t: float, vertical_scale: float, upward: bool) -> np.ndarray:
                base = blend(mouth_right, mouth_left, t)
                curve = 1.0 - ((t - 0.5) / 0.5) ** 2
                sign = -1.0 if upward else 1.0
                return base + np.array([0.0, sign * vertical_scale * curve], dtype=np.float32)

            outer_top = [
                lip_arc(0.16, 0.55 * height_scale, True),
                lip_arc(0.32, 0.78 * height_scale, True),
                lip_arc(0.50, (0.95 + 0.25 * lip_roundness) * height_scale, True),
                lip_arc(0.68, 0.78 * height_scale, True),
                lip_arc(0.84, 0.55 * height_scale, True),
            ]
            outer_bottom = [
                lip_arc(0.84, 0.48 * height_scale, False),
                lip_arc(0.68, 0.78 * height_scale, False),
                lip_arc(0.50, (1.05 + 0.35 * lip_roundness) * height_scale, False),
                lip_arc(0.32, 0.78 * height_scale, False),
                lip_arc(0.16, 0.48 * height_scale, False),
            ]
            inner_top = [
                lip_arc(0.28, 0.28 * inner_height, True),
                lip_arc(0.50, 0.42 * inner_height, True),
                lip_arc(0.72, 0.28 * inner_height, True),
            ]
            inner_bottom = [
                lip_arc(0.72, 0.34 * inner_height, False),
                lip_arc(0.50, 0.52 * inner_height, False),
                lip_arc(0.28, 0.34 * inner_height, False),
            ]

            put(FacialLandmarks.MOUTH_OUTER_RIGHT, mouth_right[0], mouth_right[1], conf)
            put(FacialLandmarks.MOUTH_OUTER_TOP_1, outer_top[0][0], outer_top[0][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_TOP_2, outer_top[1][0], outer_top[1][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_TOP_3, outer_top[2][0], outer_top[2][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_TOP_4, outer_top[3][0], outer_top[3][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_TOP_5, outer_top[4][0], outer_top[4][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_LEFT, mouth_left[0], mouth_left[1], conf)
            put(FacialLandmarks.MOUTH_OUTER_BOTTOM_1, outer_bottom[0][0], outer_bottom[0][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_BOTTOM_2, outer_bottom[1][0], outer_bottom[1][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_BOTTOM_3, outer_bottom[2][0], outer_bottom[2][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_BOTTOM_4, outer_bottom[3][0], outer_bottom[3][1], conf)
            put(FacialLandmarks.MOUTH_OUTER_BOTTOM_5, outer_bottom[4][0], outer_bottom[4][1], conf)

            put(FacialLandmarks.MOUTH_INNER_RIGHT, mouth_right[0], mouth_right[1], conf)
            put(FacialLandmarks.MOUTH_INNER_TOP_1, inner_top[0][0], inner_top[0][1], conf)
            put(FacialLandmarks.MOUTH_INNER_TOP_2, inner_top[1][0], inner_top[1][1], conf)
            put(FacialLandmarks.MOUTH_INNER_TOP_3, inner_top[2][0], inner_top[2][1], conf)
            put(FacialLandmarks.MOUTH_INNER_LEFT, mouth_left[0], mouth_left[1], conf)
            put(FacialLandmarks.MOUTH_INNER_BOTTOM_1, inner_bottom[0][0], inner_bottom[0][1], conf)
            put(FacialLandmarks.MOUTH_INNER_BOTTOM_2, inner_bottom[1][0], inner_bottom[1][1], conf)
            put(FacialLandmarks.MOUTH_INNER_BOTTOM_3, inner_bottom[2][0], inner_bottom[2][1], conf)

        if keypoints is not None and keypoints.shape == (5, 2):
            le = keypoints[0]
            re = keypoints[1]
            no = keypoints[2]
            ml = keypoints[3]
            mr = keypoints[4]

            put(45, le[0], le[1])
            put(42, le[0], le[1])
            put(69, le[0], le[1])

            put(36, re[0], re[1])
            put(39, re[0], re[1])
            put(68, re[0], re[1])

            put(30, no[0], no[1])
            mouth_center = 0.5 * (ml + mr)
            mouth_width = float(np.linalg.norm(mr - ml))
            nose_to_mouth = max(2.0, float(mouth_center[1] - no[1]))
            fallback_height = max(0.10 * mouth_width, 0.25 * nose_to_mouth)
            mouth_height = self.estimate_mouth_opening(
                frame, ml, mr, mouth_center, fallback_height
            )
            put_mouth(ml, mr, mouth_center, mouth_height)

            chin_x = (x1 + x2) * 0.5
            chin_y = y2
            put(8, chin_x, chin_y)
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            fw = max(1.0, float(x2 - x1))
            fh = max(1.0, float(y2 - y1))

            put(68, cx - fw * 0.18, cy - fh * 0.10)
            put(69, cx + fw * 0.18, cy - fh * 0.10)
            put(36, cx - fw * 0.25, cy - fh * 0.10)
            put(39, cx - fw * 0.10, cy - fh * 0.10)
            put(42, cx + fw * 0.10, cy - fh * 0.10)
            put(45, cx + fw * 0.25, cy - fh * 0.10)
            put(30, cx, cy + fh * 0.02)
            mouth_left = np.array([cx + fw * 0.14, cy + fh * 0.18], dtype=np.float32)
            mouth_right = np.array([cx - fw * 0.14, cy + fh * 0.18], dtype=np.float32)
            mouth_center = 0.5 * (mouth_left + mouth_right)
            mouth_height = self.estimate_mouth_opening(
                frame, mouth_left, mouth_right, mouth_center, 0.08 * fh
            )
            put_mouth(mouth_left, mouth_right, mouth_center, mouth_height, conf=0.7)
            put(8, cx, cy + fh * 0.34)

        self.per_face_landmarks_pubs[face_id].publish(lm)

    def compute_debug_landmark_points(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        keypoints: Optional[np.ndarray],
    ):
        points = []

        def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            return (1.0 - t) * a + t * b

        def add_point(point: np.ndarray, color):
            points.append((float(point[0]), float(point[1]), color))

        def add_mouth_points(
            mouth_left: np.ndarray,
            mouth_right: np.ndarray,
            mouth_center: np.ndarray,
            mouth_height: float,
        ):
            width_vec = mouth_left - mouth_right
            mouth_width = max(1.0, float(np.linalg.norm(width_vec)))
            height_scale = max(2.0, float(mouth_height))
            lip_roundness = np.clip(height_scale / mouth_width, 0.08, 0.55)
            inner_height = max(1.0, 0.55 * height_scale)

            def lip_arc(t: float, vertical_scale: float, upward: bool) -> np.ndarray:
                base = blend(mouth_right, mouth_left, t)
                curve = 1.0 - ((t - 0.5) / 0.5) ** 2
                sign = -1.0 if upward else 1.0
                return base + np.array([0.0, sign * vertical_scale * curve], dtype=np.float32)

            outer = [
                mouth_right,
                lip_arc(0.16, 0.55 * height_scale, True),
                lip_arc(0.32, 0.78 * height_scale, True),
                lip_arc(0.50, (0.95 + 0.25 * lip_roundness) * height_scale, True),
                lip_arc(0.68, 0.78 * height_scale, True),
                lip_arc(0.84, 0.55 * height_scale, True),
                mouth_left,
                lip_arc(0.84, 0.48 * height_scale, False),
                lip_arc(0.68, 0.78 * height_scale, False),
                lip_arc(0.50, (1.05 + 0.35 * lip_roundness) * height_scale, False),
                lip_arc(0.32, 0.78 * height_scale, False),
                lip_arc(0.16, 0.48 * height_scale, False),
            ]
            inner = [
                mouth_right,
                lip_arc(0.28, 0.28 * inner_height, True),
                lip_arc(0.50, 0.42 * inner_height, True),
                lip_arc(0.72, 0.28 * inner_height, True),
                mouth_left,
                lip_arc(0.72, 0.34 * inner_height, False),
                lip_arc(0.50, 0.52 * inner_height, False),
                lip_arc(0.28, 0.34 * inner_height, False),
            ]

            for point in outer:
                add_point(point, (0, 165, 255))
            for point in inner:
                add_point(point, (0, 255, 255))

        if keypoints is not None and keypoints.shape == (5, 2):
            for point in keypoints:
                add_point(point, (255, 255, 0))

            no = keypoints[2]
            ml = keypoints[3]
            mr = keypoints[4]
            mouth_center = 0.5 * (ml + mr)
            mouth_width = float(np.linalg.norm(mr - ml))
            nose_to_mouth = max(2.0, float(mouth_center[1] - no[1]))
            mouth_height = max(0.18 * mouth_width, 0.70 * nose_to_mouth)
            add_mouth_points(ml, mr, mouth_center, mouth_height)
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            fw = max(1.0, float(x2 - x1))
            fh = max(1.0, float(y2 - y1))

            eye_right = np.array([cx - fw * 0.18, cy - fh * 0.10], dtype=np.float32)
            eye_left = np.array([cx + fw * 0.18, cy - fh * 0.10], dtype=np.float32)
            nose = np.array([cx, cy + fh * 0.02], dtype=np.float32)
            mouth_left = np.array([cx + fw * 0.14, cy + fh * 0.18], dtype=np.float32)
            mouth_right = np.array([cx - fw * 0.14, cy + fh * 0.18], dtype=np.float32)

            for point in [eye_right, eye_left, nose, mouth_right, mouth_left]:
                add_point(point, (255, 255, 0))

            mouth_center = 0.5 * (mouth_left + mouth_right)
            add_mouth_points(mouth_left, mouth_right, mouth_center, 0.10 * fh)

        return points

    def publish_face_transforms(
        self,
        header,
        face_id: str,
        bbox_xyxy: Tuple[int, int, int, int],
        keypoints: Optional[np.ndarray],
    ):
        if self.camera_matrix is None:
            if not self.warned_missing_camera_info:
                self.get_logger().warning(
                    'Skipping face TF publication until camera_info has been received'
                )
                self.warned_missing_camera_info = True
            return

        x1, y1, x2, y2 = bbox_xyxy
        face_width_px = max(1.0, float(x2 - x1))
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx0 = float(self.camera_matrix[0, 2])
        cy0 = float(self.camera_matrix[1, 2])

        z = (fx * self.face_width_m) / face_width_px
        tx = ((cx - cx0) * z) / fx
        ty = ((cy - cy0) * z) / fy

        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        if keypoints is not None and keypoints.shape == (5, 2) and self.dist_coeffs is not None:
            # SCRFD landmark order is mirrored relative to the 3D face model
            # we use for solvePnP, so swap left/right pairs before estimating pose.
            image_points = keypoints[[1, 0, 2, 4, 3]].astype(np.float32)
            pnp_flag = cv2.SOLVEPNP_EPNP if len(image_points) < 6 else cv2.SOLVEPNP_ITERATIVE
            success, rvec, tvec = cv2.solvePnP(
                self.face_model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=pnp_flag,
            )
            if success:
                rot_mat, _ = cv2.Rodrigues(rvec)
                tf_mat = np.eye(4, dtype=np.float64)
                tf_mat[:3, :3] = rot_mat
                qx, qy, qz, qw = tf_transformations.quaternion_from_matrix(tf_mat)
                qx, qy, qz, qw = tf_transformations.quaternion_multiply(
                    [qx, qy, qz, qw],
                    self.face_rotation_correction,
                )
                tx = float(tvec[0]) * self.face_model_scale_m
                ty = float(tvec[1]) * self.face_model_scale_m
                z = float(tvec[2]) * self.face_model_scale_m
                self.warned_pose_estimation_failed = False
            elif not self.warned_pose_estimation_failed:
                self.get_logger().warning(
                    'Face pose estimation failed; publishing approximate face TF from bounding box'
                )
                self.warned_pose_estimation_failed = True

        current_translation = np.array([tx, ty, z], dtype=np.float64)
        current_quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        prev_state = self.face_tf_state.get(face_id)
        if prev_state is not None:
            prev_translation, prev_quaternion = prev_state
            current_translation = (
                (1.0 - self.tf_position_smoothing) * prev_translation +
                self.tf_position_smoothing * current_translation
            )
            current_quaternion = np.array(
                tf_transformations.quaternion_slerp(
                    prev_quaternion,
                    current_quaternion,
                    self.tf_rotation_smoothing,
                ),
                dtype=np.float64,
            )
        self.face_tf_state[face_id] = (current_translation, current_quaternion)

        face_tf = TransformStamped()
        face_tf.header = header
        face_tf.header.frame_id = self.camera_optical_frame
        face_tf.child_frame_id = f'face_{face_id}'
        face_tf.transform.translation.x = float(current_translation[0])
        face_tf.transform.translation.y = float(current_translation[1])
        face_tf.transform.translation.z = float(current_translation[2])
        face_tf.transform.rotation.x = float(current_quaternion[0])
        face_tf.transform.rotation.y = float(current_quaternion[1])
        face_tf.transform.rotation.z = float(current_quaternion[2])
        face_tf.transform.rotation.w = float(current_quaternion[3])
        self.tf_broadcaster.sendTransform(face_tf)

        gx, gy, gz, gw = self.gaze_frame_rotation
        gaze_tf = TransformStamped()
        gaze_tf.header = header
        gaze_tf.header.frame_id = f'face_{face_id}'
        gaze_tf.child_frame_id = f'gaze_{face_id}'
        gaze_tf.transform.translation.x = 0.0
        gaze_tf.transform.translation.y = 0.0
        gaze_tf.transform.translation.z = 0.0
        gaze_tf.transform.rotation.x = float(gx)
        gaze_tf.transform.rotation.y = float(gy)
        gaze_tf.transform.rotation.z = float(gz)
        gaze_tf.transform.rotation.w = float(gw)
        self.tf_broadcaster.sendTransform(gaze_tf)

    def align_face(self, frame_bgr: np.ndarray, kps: np.ndarray, size: int = 128) -> np.ndarray:
        dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        dst *= (size / 112.0)

        src = kps.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            x1 = int(np.min(src[:, 0]))
            y1 = int(np.min(src[:, 1]))
            x2 = int(np.max(src[:, 0]))
            y2 = int(np.max(src[:, 1]))
            x1 = max(0, x1 - 20)
            y1 = max(0, y1 - 20)
            x2 = min(frame_bgr.shape[1] - 1, x2 + 20)
            y2 = min(frame_bgr.shape[0] - 1, y2 + 20)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros((size, size, 3), dtype=np.uint8)
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

        aligned = cv2.warpAffine(
            frame_bgr,
            M,
            (size, size),
            borderValue=0.0,
        )
        return aligned

    def publish_debug(self, header, debug_bgr: np.ndarray):
        h, w = debug_bgr.shape[:2]
        if w != self.debug_width:
            new_h = int(h * (self.debug_width / w))
            debug_bgr = cv2.resize(debug_bgr, (self.debug_width, new_h), interpolation=cv2.INTER_LINEAR)

        ok, enc = cv2.imencode('.jpg', debug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return

        out = CompressedImage()
        out.header = header
        out.header.frame_id = self.frame_id
        out.format = 'jpeg'
        out.data = enc.tobytes()
        self.pub_debug.publish(out)

    def log_tracker_events(self):
        interesting = []
        for event in self.tracker.last_debug_events:
            if event.event_type == 'match' and event.mode == 'iou':
                continue

            if event.event_type == 'match':
                interesting.append(
                    f"track={event.track_id} det={event.det_idx} mode={event.mode} "
                    f"last_iou={event.last_iou:.2f} predicted_iou={event.predicted_iou:.2f} "
                    f"center_ratio={event.center_distance_ratio:.2f}"
                )
            elif event.event_type == 'new_track':
                interesting.append(
                    f"new_track={event.track_id} det={event.det_idx} score={event.score:.2f}"
                )
            elif event.event_type == 'deleted':
                interesting.append(f"deleted_track={event.track_id}")

        if interesting:
            self.get_logger().info("Tracker events: " + " | ".join(interesting[:8]))

    def log_fps(
        self,
        n_tracks: int,
        n_dets: int,
        decode_ms: float,
        detect_ms: float,
        track_ms: float,
        publish_ms: float,
        debug_ms: float,
    ):
        self.frame_count += 1
        self.stage_totals['decode_ms'] += decode_ms
        self.stage_totals['detect_ms'] += detect_ms
        self.stage_totals['track_ms'] += track_ms
        self.stage_totals['publish_ms'] += publish_ms
        self.stage_totals['debug_ms'] += debug_ms
        now = time.time()
        dt = now - self.last_log_time
        if dt >= 1.0:
            fps = self.frame_count / dt
            if self.performance_debug:
                avg_decode_ms = self.stage_totals['decode_ms'] / self.frame_count
                avg_detect_ms = self.stage_totals['detect_ms'] / self.frame_count
                avg_track_ms = self.stage_totals['track_ms'] / self.frame_count
                avg_publish_ms = self.stage_totals['publish_ms'] / self.frame_count
                avg_debug_ms = self.stage_totals['debug_ms'] / self.frame_count
                self.get_logger().info(
                    f'SCRFD ROS4HRI: {fps:.2f} FPS | detections={n_dets} | published_tracks={n_tracks} '
                    f'| decode={avg_decode_ms:.1f}ms detect={avg_detect_ms:.1f}ms '
                    f'track={avg_track_ms:.1f}ms publish={avg_publish_ms:.1f}ms debug={avg_debug_ms:.1f}ms'
                )
            self.frame_count = 0
            self.last_log_time = now
            for key in self.stage_totals:
                self.stage_totals[key] = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = Ros4HriFaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
