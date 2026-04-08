# asr_vosk

This repository is a PAL wrapper of offline Speech Recognition model [Vosk](https://alphacephei.com/vosk/).

## Preparation

The vosk node relies on models which are distributed in separate packages,
collected in the [vosk_language_models](https://github.com/ros4hri/asr_vosk_language_models) repository.
The related debians follow the naming scheme `pal-alum-asr-vosk-language-model-<locale>-<model_size>`.

It also depends on the `vosk` debian package, which is a packaged virtual environment containing vosk.
If you do not have it available, you can install via pip: `pip install vosk==0.3.45`.

## ROS API

### Parameters

All parameters are loaded in the lifecycle `configuration` transition.

- `audio_rate` (int, default: 16000): Device sampling rate.
- `model` (string, default: "vosk_model_small"): Model family name.
- `default_locale` (string, default: "en_US"):
  The desired default_locale, using following format:
  the [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes),
  followed by an underscore,
  followed by the [ISO 3166-1 alpha-2 region code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
- `voice_id_prefix` (string, default: "voice"):
  Prefix used to generate transient voice IDs when a new speech segment is detected.

### Topics

#### Subscribed

- `audio/channel0` ([audio_common_msgs/AudioData](https://github.com/ros-drivers/audio_common/blob/ros2/audio_common_msgs/msg/AudioData.msg)):
  Microphone audio stream.
- `audio/voice_detected` ([std_msgs/Bool](https://github.com/ros2/common_interfaces/blob/humble/std_msgs/msg/Bool.msg)):
  Microphone voice activation detection.
- `/robot_speaking` ([std_msgs/Bool](https://github.com/ros2/common_interfaces/blob/humble/std_msgs/msg/Bool.msg)):
  (QoS: transient local).
  While the robot is speakin, no message is published on `/humans/voices/*`.

#### Published

- `/humans/voices/tracked` ([hri_msgs/IdsList](https://github.com/ros4hri/hri_msgs/blob/humble-devel/msg/IdsList.msg)):
  List of tracked transient voice IDs, for example `voice_0001`.
- `/humans/voices/<voice_id>/audio` ([audio_common_msgs/AudioData](https://github.com/ros-drivers/audio_common/blob/ros2/audio_common_msgs/msg/AudioData.msg)):
  Voice audio stream.
- `/humans/voices/<voice_id>/is_speaking` ([std_msgs/Bool](https://github.com/ros2/common_interfaces/blob/humble/std_msgs/msg/Bool.msg)):
  Voice speech detection.
- `/humans/voices/<voice_id>/speech` ([hri_msgs/LiveSpeech](https://github.com/ros4hri/hri_msgs/blob/humble-devel/msg/LiveSpeech.msg)):
  Speech recognized.
- `/diagnostics` ([diagnostic_msgs/DiagnosticArray](https://github.com/ros2/common_interfaces/blob/humble/diagnostic_msgs/msg/DiagnosticArray.msg))

### Services

#### Servers

- `~/get_supported_locales` ([i18n_msgs/GetLocales](https://gitlab/interaction/i18n_msgs/-/blob/humble-devel/srv/GetLocales.srv)):
  Get the list of locales supported and configurable in the `default_locale` parameter.
  It is computed at runtime during configure transition from the list of installed models found for `model` parameter.

### Actions

#### Servers

- `~/set_default_locale` ([i18n_msgs/SetLocale](https://gitlab/interaction/i18n_msgs/-/blob/humble-devel/action/SetLocale.action)):
  Sets the `default_locale` parameter and loads the corresponding model.

## Resources

(For an intro on resources, see [ament_index](https://github.com/ament/ament_cmake/blob/master/ament_cmake_core/doc/resource_index.md)).

- `asr.vosk.models`:
  All the available models are installed under this resource.
  It expects marker files containing the `model descriptor` path, separated in different lines, relative to its package share folder install path.
  The `model descriptor` is a YAML file containing:
  - model family name,
  - locale,
  - model binary location.
  Various `asr_vosk_language_model_<locale>_<size>` packages install each one such model.

## Launch


```bash
ros2 launch asr_vosk asr_vosk.launch.py
```

The `asr_vosk.launch.py` launch file accepts as arguments and configures the defined [parameters](#parameters).
It also automatically transitions the node to the active state.

## Example

To test the package using the system default microphone:

1. Install the
   [`audio_capture`](https://github.com/ros-drivers/audio_common/tree/ros2)
   package (`sudo apt install pal-alum-audio-capture` on PAL robots)
1. Launch the `audio_capture` package:
   `ros2 launch audio_capture capture.launch.xml audio_topic:=channel0 format:=wave`
1. In a new terminal, launch the `asr_vosk` package:
   `ros2 launch asr_vosk asr_vosk.launch.py`
1. Check the currently tracked voice ID:
   `ros2 topic echo /humans/voices/tracked`
1. Then inspect the recognized speech output for that voice, for example:
   `ros2 topic echo /humans/voices/voice_0001/speech`
