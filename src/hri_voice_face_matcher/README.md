# hri_voice_face_matcher

`hri_voice_face_matcher` associates tracked voices with tracked persons in a
ROS4HRI pipeline and publishes the result as `hri_msgs/IdsMatch` messages on
`/humans/candidate_matches`.

The current implementation uses two cues:

- if exactly one non-anonymous tracked person is `ENGAGED`, the active voice is
  associated to that person with high confidence
- if several engaged people are present, the matcher compares lower-face motion
  from `/humans/faces/<face_id>/aligned` or `/cropped` images and associates the
  active voice to the person whose face is most active

The published match is `VOICE -> PERSON`, which is directly consumable by
`hri_person_manager` and remains compliant with REP-155.

## Inputs

- `/humans/voices/tracked`
- `/humans/voices/<voice_id>/is_speaking`
- `/humans/persons/tracked`
- `/humans/persons/<person_id>/face_id`
- `/humans/persons/<person_id>/engagement_status`
- `/humans/faces/<face_id>/aligned`
- `/humans/faces/<face_id>/cropped`

## Output

- `/humans/candidate_matches`

## Notes

This is a heuristic matcher. It works best when:

- the ASR pipeline exposes one active voice at a time
- `hri_face_identification`, `hri_engagement`, and `hri_person_manager` are already running
- aligned face images are published by the face detector
