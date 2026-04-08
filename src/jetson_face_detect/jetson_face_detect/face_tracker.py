#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .scrfd_trt_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    keypoints: Optional[np.ndarray] = None
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    age: int = 0
    hits: int = 1
    missed: int = 0


@dataclass
class TrackerDebugEvent:
    event_type: str
    track_id: int
    det_idx: Optional[int] = None
    mode: Optional[str] = None
    score: Optional[float] = None
    last_iou: Optional[float] = None
    predicted_iou: Optional[float] = None
    center_distance_ratio: Optional[float] = None
    missed: Optional[int] = None


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_size(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1)))


class SimpleFaceTracker:
    def __init__(
        self,
        iou_threshold: float = 0.35,
        min_predicted_iou: float = 0.05,
        max_center_distance_ratio: float = 0.9,
        max_missed: int = 6,
        min_hits: int = 1,
        debug: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.min_predicted_iou = min_predicted_iou
        self.max_center_distance_ratio = max_center_distance_ratio
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.debug = debug
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.last_debug_events: List[TrackerDebugEvent] = []

    def predict_bbox(self, track: Track) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = track.bbox_xyxy
        vx, vy = track.velocity_xy
        return (
            int(round(x1 + vx)),
            int(round(y1 + vy)),
            int(round(x2 + vx)),
            int(round(y2 + vy)),
        )

    def normalized_center_distance(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        ax, ay = bbox_center(a)
        bx, by = bbox_center(b)
        aw, ah = bbox_size(a)
        bw, bh = bbox_size(b)
        norm_w = max(aw, bw)
        norm_h = max(ah, bh)
        dx = (ax - bx) / norm_w
        dy = (ay - by) / norm_h
        return float(np.hypot(dx, dy))

    def compute_match_details(self, track: Track, det: Detection):
        last_iou = iou_xyxy(track.bbox_xyxy, det.bbox_xyxy)
        predicted_bbox = self.predict_bbox(track)
        predicted_iou = iou_xyxy(predicted_bbox, det.bbox_xyxy)
        center_distance_ratio = self.normalized_center_distance(predicted_bbox, det.bbox_xyxy)

        if last_iou >= self.iou_threshold:
            return {
                "score": 2.0 + last_iou,
                "mode": "iou",
                "last_iou": last_iou,
                "predicted_iou": predicted_iou,
                "center_distance_ratio": center_distance_ratio,
            }

        if predicted_iou < self.min_predicted_iou:
            return {
                "score": -1.0,
                "mode": "reject_low_predicted_iou",
                "last_iou": last_iou,
                "predicted_iou": predicted_iou,
                "center_distance_ratio": center_distance_ratio,
            }

        if center_distance_ratio > self.max_center_distance_ratio:
            return {
                "score": -1.0,
                "mode": "reject_far_center",
                "last_iou": last_iou,
                "predicted_iou": predicted_iou,
                "center_distance_ratio": center_distance_ratio,
            }

        return {
            "score": predicted_iou + max(0.0, 1.0 - center_distance_ratio),
            "mode": "predicted",
            "last_iou": last_iou,
            "predicted_iou": predicted_iou,
            "center_distance_ratio": center_distance_ratio,
        }

    def update(self, detections: List[Detection]) -> List[Track]:
        self.last_debug_events = []
        matched_track_ids = set()
        matched_det_indices = set()
        for det_idx, det in enumerate(detections):
            best_track_id = None
            best_score = -1.0
            best_details = None

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                details = self.compute_match_details(track, det)
                if details["score"] > best_score:
                    best_score = details["score"]
                    best_track_id = track_id
                    best_details = details

            if best_track_id is not None and best_score >= 0.0:
                track = self.tracks[best_track_id]
                prev_center_x, prev_center_y = bbox_center(track.bbox_xyxy)
                new_center_x, new_center_y = bbox_center(det.bbox_xyxy)
                measured_velocity = (
                    new_center_x - prev_center_x,
                    new_center_y - prev_center_y,
                )
                track.bbox_xyxy = det.bbox_xyxy
                track.score = det.score
                track.keypoints = det.keypoints
                track.velocity_xy = (
                    0.6 * track.velocity_xy[0] + 0.4 * measured_velocity[0],
                    0.6 * track.velocity_xy[1] + 0.4 * measured_velocity[1],
                )
                track.age += 1
                track.hits += 1
                track.missed = 0

                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)

                if self.debug and best_details is not None:
                    self.last_debug_events.append(
                        TrackerDebugEvent(
                            event_type="match",
                            track_id=best_track_id,
                            det_idx=det_idx,
                            mode=best_details["mode"],
                            score=float(best_details["score"]),
                            last_iou=float(best_details["last_iou"]),
                            predicted_iou=float(best_details["predicted_iou"]),
                            center_distance_ratio=float(best_details["center_distance_ratio"]),
                            missed=track.missed,
                        )
                    )

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue

            track = Track(
                track_id=self.next_track_id,
                bbox_xyxy=det.bbox_xyxy,
                score=det.score,
                keypoints=det.keypoints,
                age=1,
                hits=1,
                missed=0,
            )
            self.tracks[self.next_track_id] = track
            if self.debug:
                self.last_debug_events.append(
                    TrackerDebugEvent(
                        event_type="new_track",
                        track_id=self.next_track_id,
                        det_idx=det_idx,
                        score=float(det.score),
                    )
                )
            matched_track_ids.add(self.next_track_id)
            self.next_track_id += 1

        to_delete = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.age += 1
                track.missed += 1
                if self.debug:
                    self.last_debug_events.append(
                        TrackerDebugEvent(
                            event_type="missed",
                            track_id=track_id,
                            missed=track.missed,
                        )
                    )
                if track.missed > self.max_missed:
                    to_delete.append(track_id)

        for track_id in to_delete:
            if self.debug:
                self.last_debug_events.append(
                    TrackerDebugEvent(
                        event_type="deleted",
                        track_id=track_id,
                    )
                )
            del self.tracks[track_id]

        active = [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits and t.missed <= self.max_missed
        ]
        active.sort(key=lambda t: t.track_id)
        return active

    def predict_only(self) -> List[Track]:
        self.last_debug_events = []
        to_delete = []
        for track_id, track in self.tracks.items():
            track.bbox_xyxy = self.predict_bbox(track)
            track.age += 1
            track.missed += 1
            if self.debug:
                self.last_debug_events.append(
                    TrackerDebugEvent(
                        event_type="missed",
                        track_id=track_id,
                        missed=track.missed,
                    )
                )
            if track.missed > self.max_missed:
                to_delete.append(track_id)

        for track_id in to_delete:
            if self.debug:
                self.last_debug_events.append(
                    TrackerDebugEvent(
                        event_type="deleted",
                        track_id=track_id,
                    )
                )
            del self.tracks[track_id]

        active = [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits and t.missed <= self.max_missed
        ]
        active.sort(key=lambda t: t.track_id)
        return active
