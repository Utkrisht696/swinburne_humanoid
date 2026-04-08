#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
from hri import HRIListener
from hri_msgs.msg import EngagementLevel, IdsMatch
import numpy as np
import rclpy
from rclpy.node import Node


@dataclass
class FaceActivityState:
    signature: Optional[np.ndarray] = None
    motion_score: float = 0.0


@dataclass
class CandidateScore:
    person_id: str
    confidence: float
    reason: str


class VoiceFaceMatcherNode(Node):
    def __init__(self):
        super().__init__('hri_voice_face_matcher')

        self.declare_parameter('rate', 10.0)
        self.declare_parameter('log_decisions', True)
        self.declare_parameter('allow_single_tracked_voice_fallback', True)
        self.declare_parameter('confidence_threshold', 0.55)
        self.declare_parameter('single_identified_engaged_confidence', 0.95)
        self.declare_parameter('single_engaged_confidence', 0.75)
        self.declare_parameter('single_identified_confidence', 0.78)
        self.declare_parameter('single_tracked_confidence', 0.68)
        self.declare_parameter('motion_base_confidence', 0.65)
        self.declare_parameter('motion_gain', 0.8)
        self.declare_parameter('motion_margin_gain', 1.2)
        self.declare_parameter('motion_threshold', 0.02)
        self.declare_parameter('signature_size', 48)

        self.rate = float(self.get_parameter('rate').value)
        self.log_decisions = bool(self.get_parameter('log_decisions').value)
        self.allow_single_tracked_voice_fallback = bool(
            self.get_parameter('allow_single_tracked_voice_fallback').value
        )
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.single_identified_engaged_confidence = float(
            self.get_parameter('single_identified_engaged_confidence').value
        )
        self.single_engaged_confidence = float(
            self.get_parameter('single_engaged_confidence').value
        )
        self.single_identified_confidence = float(
            self.get_parameter('single_identified_confidence').value
        )
        self.single_tracked_confidence = float(
            self.get_parameter('single_tracked_confidence').value
        )
        self.motion_base_confidence = float(self.get_parameter('motion_base_confidence').value)
        self.motion_gain = float(self.get_parameter('motion_gain').value)
        self.motion_margin_gain = float(self.get_parameter('motion_margin_gain').value)
        self.motion_threshold = float(self.get_parameter('motion_threshold').value)
        self.signature_size = int(self.get_parameter('signature_size').value)

        self.hri_listener = HRIListener('hri_voice_face_matcher_listener')
        self.match_pub = self.create_publisher(IdsMatch, '/humans/candidate_matches', 10)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

        self.face_activity: Dict[str, FaceActivityState] = {}
        self.last_published: Dict[str, Tuple[str, str]] = {}
        self.last_status: Optional[str] = None
        self.last_candidate_hint: Optional[str] = None

        self.get_logger().info(
            'Matching voices to persons using engagement and lower-face motion cues'
        )

    def on_timer(self):
        voices = list(self.hri_listener.voices.items())
        speaking_voices = [(voice_id, voice) for voice_id, voice in voices if voice.is_speaking]

        if not voices:
            self.log_status('waiting: no voices tracked')
            return

        if not speaking_voices:
            if self.allow_single_tracked_voice_fallback and len(voices) == 1:
                speaking_voices = voices
                self.log_status(
                    f'fallback: using single tracked voice {voices[0][0]} without speaking flag'
                )
            else:
                tracked_people = len(self.hri_listener.tracked_persons)
                self.log_status(f'waiting: no speaking voices (tracked_persons={tracked_people})')
                return

        for voice_id, _voice in speaking_voices:
            candidate = self.select_person_for_voice()
            if candidate is None or candidate.confidence < self.confidence_threshold:
                if candidate is None:
                    self.log_status(
                        f'skipping {voice_id}: no suitable person candidate '
                        f'(tracked_persons={len(self.hri_listener.tracked_persons)})'
                    )
                else:
                    self.log_status(
                        f'skipping {voice_id}: candidate {candidate.person_id} below threshold '
                        f'({candidate.confidence:.2f} < {self.confidence_threshold:.2f}, '
                        f'reason={candidate.reason})'
                    )
                continue

            previous = self.last_published.get(voice_id)
            current_state = (candidate.person_id, candidate.reason)
            if previous == current_state:
                self.log_status(
                    f'unchanged {voice_id} -> {candidate.person_id} ({candidate.reason})'
                )
                continue

            self.publish_match(voice_id, candidate)
            self.last_published[voice_id] = current_state
            self.log_status(
                f'publishing {voice_id} -> {candidate.person_id} '
                f'({candidate.reason}, confidence={candidate.confidence:.2f})'
            )

    def log_status(self, status: str):
        if not self.log_decisions:
            return
        if status == self.last_status:
            return
        self.last_status = status
        self.get_logger().info(status)

    def update_face_activity(self):
        active_face_ids = set()

        for face_id, face in list(self.hri_listener.faces.items()):
            active_face_ids.add(face_id)

            image = face.aligned if face.aligned is not None else face.cropped
            if image is None:
                continue

            try:
                signature = self.compute_lower_face_signature(image)
            except Exception as exc:
                self.log_status(f'warning: failed to read face image for {face_id}: {exc}')
                continue
            state = self.face_activity.setdefault(face_id, FaceActivityState())

            if state.signature is not None:
                delta = cv2.absdiff(signature, state.signature)
                motion = float(np.mean(delta)) / 255.0
                state.motion_score = 0.6 * state.motion_score + 0.4 * motion

            state.signature = signature

        for face_id in list(self.face_activity.keys()):
            if face_id not in active_face_ids:
                del self.face_activity[face_id]

    def compute_lower_face_signature(self, image: np.ndarray) -> np.ndarray:
        height = image.shape[0]
        lower_half = image[height // 2:, :]
        gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            gray,
            (self.signature_size, self.signature_size),
            interpolation=cv2.INTER_AREA,
        )

    def select_person_for_voice(self) -> Optional[CandidateScore]:
        candidates = self.get_engaged_candidates()
        identified = [person for person in candidates if person.anonymous is False]

        if len(identified) == 1:
            self.log_candidate_hint(
                f'candidate: single identified engaged person {identified[0].id}'
            )
            return CandidateScore(
                person_id=identified[0].id,
                confidence=self.single_identified_engaged_confidence,
                reason='single_identified_engaged_person',
            )

        if len(candidates) == 1:
            self.log_candidate_hint(
                f'candidate: single engaged person {candidates[0].id}'
            )
            return CandidateScore(
                person_id=candidates[0].id,
                confidence=self.single_engaged_confidence,
                reason='single_engaged_person',
            )

        if candidates:
            motion_pool = identified if len(identified) >= 2 else candidates
            return self.select_by_face_motion(motion_pool, 'face_motion_engaged')

        tracked_candidates = self.get_tracked_candidates()
        tracked_identified = [person for person in tracked_candidates if person.anonymous is False]

        if len(tracked_identified) == 1:
            self.log_candidate_hint(
                f'candidate: single identified tracked person {tracked_identified[0].id}'
            )
            return CandidateScore(
                person_id=tracked_identified[0].id,
                confidence=self.single_identified_confidence,
                reason='single_identified_person',
            )

        if len(tracked_candidates) == 1:
            self.log_candidate_hint(
                f'candidate: single tracked person {tracked_candidates[0].id}'
            )
            return CandidateScore(
                person_id=tracked_candidates[0].id,
                confidence=self.single_tracked_confidence,
                reason='single_tracked_person',
            )

        if tracked_candidates:
            motion_pool = tracked_identified if len(tracked_identified) >= 2 else tracked_candidates
            return self.select_by_face_motion(motion_pool, 'face_motion_tracked')

        return None

    def log_candidate_hint(self, status: str):
        if not self.log_decisions:
            return
        if status == self.last_candidate_hint:
            return
        self.last_candidate_hint = status
        self.get_logger().info(status)

    def get_engaged_candidates(self):
        candidates = []
        for _, person in list(self.hri_listener.tracked_persons.items()):
            if person.face is None:
                continue
            if person.engagement_status != EngagementLevel.ENGAGED:
                continue
            candidates.append(person)
        return candidates

    def get_tracked_candidates(self):
        candidates = []
        for _, person in list(self.hri_listener.tracked_persons.items()):
            if person.face is None:
                continue
            candidates.append(person)
        return candidates

    def select_by_face_motion(self, persons, reason: str) -> Optional[CandidateScore]:
        self.update_face_activity()

        scored: List[Tuple[float, object]] = []
        for person in persons:
            face = person.face
            if face is None:
                continue
            score = self.face_activity.get(face.id, FaceActivityState()).motion_score
            scored.append((score, person))

        if not scored:
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_person = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < self.motion_threshold:
            return None

        margin = max(0.0, best_score - second_score)
        confidence = min(
            0.99,
            self.motion_base_confidence
            + self.motion_gain * best_score
            + self.motion_margin_gain * margin,
        )

        return CandidateScore(
            person_id=best_person.id,
            confidence=confidence,
            reason=reason,
        )

    def publish_match(self, voice_id: str, candidate: CandidateScore):
        msg = IdsMatch()
        msg.id1 = voice_id
        msg.id1_type = IdsMatch.VOICE
        msg.id2 = candidate.person_id
        msg.id2_type = IdsMatch.PERSON
        msg.confidence = float(candidate.confidence)
        self.match_pub.publish(msg)

        self.get_logger().info(
            f'Matched voice {voice_id} to person {candidate.person_id} '
            f'with confidence {candidate.confidence:.2f} ({candidate.reason})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VoiceFaceMatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
