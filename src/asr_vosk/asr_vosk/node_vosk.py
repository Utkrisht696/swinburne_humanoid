# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_index_python.packages import get_package_share_path
from ament_index_python.resources import get_resource, get_resources
from array import array
from audio_common_msgs.msg import AudioData, AudioStamped
from collections import defaultdict
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_msgs.msg import IdsList, LiveSpeech
from i18n_msgs.action import SetLocale
from i18n_msgs.srv import GetLocales
import json
from lifecycle_msgs.msg import State
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.lifecycle import Node, LifecycleState, TransitionCallbackReturn
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from vosk import Model, KaldiRecognizer
import yaml


class NodeVosk(Node):
    """Vosk speech recognition node."""

    def __init__(self):
        super().__init__('asr_vosk')

        self.declare_parameter(
            'audio_rate', 16000, ParameterDescriptor(description='Device sampling rate'))
        self.declare_parameter(
            'model', "vosk_model_small", ParameterDescriptor(description='Model family name'))
        self.declare_parameter(
            'default_locale', "en_US", ParameterDescriptor(description='Default locale'))
        self.declare_parameter(
            'voice_id_prefix', "voice",
            ParameterDescriptor(description='Prefix used for generated transient voice IDs'))
        self.declare_parameter(
            'voice_silence_timeout_sec', 2.0,
            ParameterDescriptor(
                description='How long to keep the current voice ID alive after speech stops'))
        self.declare_parameter(
            'audio_stamped_topic', "/audio",
            ParameterDescriptor(description='Primary audio topic using audio_common_msgs/AudioStamped'))
        self.declare_parameter(
            'legacy_audio_topic', "audio/channel0",
            ParameterDescriptor(description='Legacy audio topic using audio_common_msgs/AudioData'))
        self.declare_parameter(
            'use_legacy_audio_topic', True,
            ParameterDescriptor(description='Keep subscribing to the legacy AudioData topic'))

        self.get_logger().info('State: Unconfigured.')

    def __del__(self):
        self.trigger_shutdown()

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_cleanup()
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.audio_rate = self.get_parameter('audio_rate').value
        self.default_locale = self.get_parameter('default_locale').value
        self.model = self.get_parameter('model').value
        self.voice_id_prefix = self.get_parameter('voice_id_prefix').value
        self.voice_silence_timeout_sec = float(
            self.get_parameter('voice_silence_timeout_sec').value)
        self.audio_stamped_topic = self.get_parameter('audio_stamped_topic').value
        self.legacy_audio_topic = self.get_parameter('legacy_audio_topic').value
        self.use_legacy_audio_topic = bool(self.get_parameter('use_legacy_audio_topic').value)

        # Load available models and set the supported locales accordingly
        resource_type = 'asr.vosk.model'
        self.available_models = defaultdict(dict)
        for pkg in get_resources(resource_type).keys():
            pkg_share_path = get_package_share_path(pkg)
            cfg_file_path = pkg_share_path / get_resource(resource_type, pkg)[0]
            with open(cfg_file_path, 'r') as f:
                cfg = yaml.safe_load(f)
                try:
                    if self.model == cfg['name']:
                        self.available_models[cfg['locale']] = pkg_share_path / cfg['path']
                except KeyError as e:
                    self.get_logger().error(
                        f'Error parsing configuration for package {pkg}: {str(e)}')
                    return TransitionCallbackReturn.FAILURE

        loaded_model, _ = self.load_model(self.default_locale)
        if not loaded_model:
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info('State: Inactive.')
        return super().on_configure(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_deactivate()
        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.current_incremental = ''
        self.last_final = ''
        self.listening = True
        self.voice_counter = 0
        self.current_voice_id = None
        self.current_is_speaking = False
        self.last_voice_activity_time = None
        self.voice_publishers = {}

        self.diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 1)
        self.voices_pub = self.create_publisher(
            IdsList, "/humans/voices/tracked", 1)

        self.audio_stamped_sub = self.create_subscription(
            AudioStamped, self.audio_stamped_topic, self.on_audio_stamped, 10)
        self.audio_data_sub = None
        if self.use_legacy_audio_topic:
            self.audio_data_sub = self.create_subscription(
                AudioData, self.legacy_audio_topic, self.on_audio_data, 10)
        self.voice_detected_sub = self.create_subscription(
            Bool, "audio/voice_detected", self.on_voice_detected, 1)
        self.robot_speaking_sub = self.create_subscription(
            Bool, "/robot_speaking", self.on_robot_speaking, 1)

        self.get_supported_locales_server = self.create_service(
            GetLocales, "~/get_supported_locales", self.on_get_supported_locales)

        self.set_default_locale_server = ActionServer(
            self, SetLocale, "~/set_default_locale",
            goal_callback=self.on_set_default_locale_goal,
            execute_callback=self.on_set_default_locale_exec)

        self.diag_timer = self.create_timer(1., self.publish_diagnostics)

        # Publish the current tracked voice ID repeatedly to ensure latecomers receive it.
        self.voices_timer = self.create_timer(
            1., self.publish_tracked_voice_ids, clock=self.get_clock())
        self.voice_timeout_timer = self.create_timer(
            0.2, self.prune_inactive_voice, clock=self.get_clock())
        self.publish_tracked_voice_ids()

        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.state_id == State.PRIMARY_STATE_ACTIVE:
            self.internal_deactivate()
        if state.state_id in [State.PRIMARY_STATE_ACTIVE, State.PRIMARY_STATE_INACTIVE]:
            self.internal_cleanup()
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def internal_cleanup(self):
        del self.recognizer

    def internal_deactivate(self):
        self.destroy_timer(self.diag_timer)
        del self.set_default_locale_server
        self.destroy_service(self.get_supported_locales_server)
        self.destroy_subscription(self.audio_stamped_sub)
        if self.audio_data_sub is not None:
            self.destroy_subscription(self.audio_data_sub)
        self.destroy_subscription(self.voice_detected_sub)
        self.destroy_subscription(self.robot_speaking_sub)
        self.destroy_publisher(self.diag_pub)
        self.destroy_publisher(self.voices_pub)
        self.destroy_timer(self.voices_timer)
        self.destroy_timer(self.voice_timeout_timer)

        for pubs in self.voice_publishers.values():
            self.destroy_publisher(pubs["speech"])
            self.destroy_publisher(pubs["audio"])
            self.destroy_publisher(pubs["is_speaking"])
        self.voice_publishers.clear()

    def load_model(self, locale):
        if locale not in self.available_models:
            error_msg = (
                f'No installed Vosk model found for model family "{self.model}" '
                f'and locale "{locale}".'
            )
            self.get_logger().error(error_msg)
            return False, error_msg

        try:
            model_path = str(self.available_models[locale])
            model = Model(model_path)
            self.get_logger().info(f'Loaded {self.model} {locale} model')
        except Exception as e:  # vosk Model raises generic exceptions :/
            error_msg = f'Failed to load {self.model} {locale} model: {str(e)}'
            self.get_logger().error(error_msg)
            return False, error_msg

        self.recognizer = KaldiRecognizer(model, self.audio_rate)
        return True, ""

    def new_voice_id(self):
        self.voice_counter += 1
        return f'{self.voice_id_prefix}_{self.voice_counter:04d}'

    def ensure_voice_publishers(self, voice_id):
        if voice_id not in self.voice_publishers:
            self.voice_publishers[voice_id] = {
                "speech": self.create_publisher(
                    LiveSpeech, f"/humans/voices/{voice_id}/speech", 10),
                "audio": self.create_publisher(
                    AudioData, f"/humans/voices/{voice_id}/audio", 10),
                "is_speaking": self.create_publisher(
                    Bool, f"/humans/voices/{voice_id}/is_speaking", 10),
            }
        return self.voice_publishers[voice_id]

    def destroy_voice_publishers(self, voice_id):
        pubs = self.voice_publishers.pop(voice_id, None)
        if pubs is None:
            return
        self.destroy_publisher(pubs["speech"])
        self.destroy_publisher(pubs["audio"])
        self.destroy_publisher(pubs["is_speaking"])

    def publish_tracked_voice_ids(self):
        ids = [self.current_voice_id] if self.current_voice_id is not None else []
        self.voices_pub.publish(IdsList(ids=ids))

    def set_current_voice(self, voice_id):
        if voice_id == self.current_voice_id:
            return

        previous_voice_id = self.current_voice_id
        if previous_voice_id is not None and previous_voice_id in self.voice_publishers:
            self.voice_publishers[previous_voice_id]["is_speaking"].publish(Bool(data=False))

        self.current_voice_id = voice_id
        if voice_id is not None:
            self.ensure_voice_publishers(voice_id)
            self.last_voice_activity_time = self.get_clock().now()
        else:
            self.last_voice_activity_time = None
        self.publish_tracked_voice_ids()

        if previous_voice_id is not None and previous_voice_id != voice_id:
            self.destroy_voice_publishers(previous_voice_id)

    def touch_current_voice(self):
        if self.current_voice_id is not None:
            self.last_voice_activity_time = self.get_clock().now()

    def prune_inactive_voice(self):
        if self.current_voice_id is None:
            return
        if self.current_is_speaking:
            return
        if self.last_voice_activity_time is None:
            return

        silence_duration = (
            self.get_clock().now() - self.last_voice_activity_time
        ).nanoseconds / 1e9
        if silence_duration >= self.voice_silence_timeout_sec:
            self.set_current_voice(None)

    def on_voice_detected(self, msg):
        speaking = bool(msg.data)

        if speaking and not self.current_is_speaking:
            if self.current_voice_id is None:
                self.set_current_voice(self.new_voice_id())
            else:
                self.touch_current_voice()

        self.current_is_speaking = speaking

        if self.current_voice_id is not None:
            if speaking:
                self.touch_current_voice()
            pubs = self.ensure_voice_publishers(self.current_voice_id)
            pubs["is_speaking"].publish(Bool(data=speaking))

    def on_robot_speaking(self, msg):
        self.listening = not msg.data

    def audio_data_to_bytes(self, audio_data_msg):
        if audio_data_msg.int16_data:
            return array('h', audio_data_msg.int16_data).tobytes()
        if audio_data_msg.uint8_data:
            return bytes(audio_data_msg.uint8_data)
        if audio_data_msg.int8_data:
            return array('b', audio_data_msg.int8_data).tobytes()
        if audio_data_msg.int32_data:
            return array('i', audio_data_msg.int32_data).tobytes()
        if audio_data_msg.float32_data:
            return array('f', audio_data_msg.float32_data).tobytes()
        return b''

    def audio_data_has_signal(self, audio_data_msg):
        for samples in (
            audio_data_msg.int16_data,
            audio_data_msg.uint8_data,
            audio_data_msg.int8_data,
            audio_data_msg.int32_data,
            audio_data_msg.float32_data,
        ):
            for sample in samples:
                if sample != 0:
                    return True
        return False

    def process_audio_data(self, audio_data_msg):
        if self.current_voice_id is not None:
            self.touch_current_voice()
            pubs = self.ensure_voice_publishers(self.current_voice_id)
            pubs["audio"].publish(audio_data_msg)

        if self.listening:
            speech_msg = LiveSpeech(locale=self.default_locale, confidence=1.)
            speech_msg.header.stamp = self.get_clock().now().to_msg()
            audio_bytes = self.audio_data_to_bytes(audio_data_msg)
            non_empty_audio_data = self.audio_data_has_signal(audio_data_msg)
            if non_empty_audio_data or self.current_incremental:
                if self.recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(self.recognizer.Result())
                    text = result["text"].strip()

                    if text:
                        speech_msg.incremental = text
                        speech_msg.final = text
                        if self.current_voice_id is None:
                            self.set_current_voice(self.new_voice_id())
                        pubs = self.ensure_voice_publishers(self.current_voice_id)
                        pubs["speech"].publish(speech_msg)
                        self.touch_current_voice()

                        self.last_final = text

                    self.current_incremental = ''
                else:
                    result = json.loads(self.recognizer.PartialResult())
                    partial = result["partial"]

                    if partial and (partial != self.current_incremental):
                        speech_msg.incremental = partial
                        if self.current_voice_id is None:
                            self.set_current_voice(self.new_voice_id())
                        pubs = self.ensure_voice_publishers(self.current_voice_id)
                        pubs["speech"].publish(speech_msg)
                        self.touch_current_voice()

                    self.current_incremental = partial

    def on_audio_stamped(self, audio_msg):
        if audio_msg.audio.info.rate > 0 and audio_msg.audio.info.rate != self.audio_rate:
            self.get_logger().warn(
                f'Received audio rate {audio_msg.audio.info.rate}Hz on {self.audio_stamped_topic}, '
                f'but recognizer is configured for {self.audio_rate}Hz'
            )
        self.process_audio_data(audio_msg.audio.audio_data)

    def on_audio_data(self, audio_data_msg):
        self.process_audio_data(audio_data_msg)

    def on_get_supported_locales(self, request, response):
        response.locales = list(self.available_models.keys())
        return response

    def on_set_default_locale_goal(self, goal_request):
        if goal_request.locale in self.available_models:
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def on_set_default_locale_exec(self, goal_handle):
        locale = goal_handle.request.locale
        result = SetLocale.Result()
        loaded_model, error_msg = self.load_model(locale)
        if loaded_model:
            self.set_parameters([Parameter('default_locale', value=locale)])
            self.default_locale = locale
            goal_handle.succeed()
        else:
            goal_handle.abort()
            result.error_msg = error_msg
        return result

    def publish_diagnostics(self):
        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name="/communication/asr/asr_vosk",
            message="vosk ASR running",
            values=[
                KeyValue(key="Module name", value="asr_vosk"),
                KeyValue(key="Current lifecycle state",
                         value=self._state_machine.current_state[1]),
                KeyValue(key="Model", value=self.model),
                KeyValue(key="Supported locales", value=str(self.available_models.keys())),
                KeyValue(key="Current default_locale", value=self.default_locale),
                KeyValue(key="Currently listening", value=str(self.listening)),
                KeyValue(key="Current voice id", value=str(self.current_voice_id)),
                KeyValue(key="Last recognised sentence", value=self.last_final),
            ],
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = NodeVosk()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
