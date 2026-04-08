# Swinburne ROS 2 Workspace

This repository is a clean GitHub-ready copy of the Swinburne ROS 2 workspace from `/home/agi/swinburne`.

It keeps the source packages and excludes generated workspace artifacts such as `build/`, `install/`, and `log/`.

## Workspace Layout

- `src/`: ROS 2 packages included in this workspace

## Included Packages

- `asr_vosk`
- `asr_vosk_language_model_en_us_gigaspeech`
- `asr_vosk_language_model_en_us_small`
- `audio_common`
- `audio_common_msgs`
- `hri`
- `hri_actions_msgs`
- `hri_body_detect`
- `hri_engagement`
- `hri_face_body_matcher`
- `hri_face_detect`
- `hri_face_identification`
- `hri_msgs`
- `hri_person_manager`
- `hri_privacy_msgs`
- `hri_voice_face_matcher`
- `human_description`
- `i18n_msgs`
- `jetson_face_detect`
- `pyhri`
- `x2_description`

## Notes

- Nested `.git` directories from the original package clones were excluded so this repository can track all package contents as a single monorepo.
- If you want to build the workspace after cloning:

```bash
cd /path/to/repo
colcon build
```
