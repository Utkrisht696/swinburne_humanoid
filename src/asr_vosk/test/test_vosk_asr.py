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

from audio_common_msgs.msg import AudioData
from datetime import datetime, timedelta
from hri import HRIListener
from pathlib import Path
import rclpy
from rclpy.executors import Executor, SingleThreadedExecutor, TimeoutException
from rclpy.parameter import Parameter
from rclpy.time import Time
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool
import unittest
from asr_vosk.pal_node_vosk import NodeVosk


def spin_some(executor: Executor, timeout=timedelta(seconds=10.)):
    start = datetime.now()
    # get first available task without waiting
    cb_iter = executor._wait_for_ready_callbacks(timeout_sec=0.)
    while True:
        try:
            handler, *_ = next(cb_iter)
            handler()
            if handler.exception() is not None:
                raise handler.exception()
        except TimeoutException:
            elapsed = datetime.now() - start
            if elapsed > timeout:
                raise TimeoutException(f'Time elapsed spinning {elapsed} with timeout {timeout}')
        except StopIteration:
            break


class TestVoskMixin():
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.vosk_node = NodeVosk()
        cls.vosk_node.set_parameters([
            Parameter(name='model', value=cls.model),
            Parameter(name='default_locale', value=cls.locale),
            Parameter(name='use_sim_time', value=True)])
        cls.vosk_executor = SingleThreadedExecutor()
        cls.vosk_executor.add_node(cls.vosk_node)
        cls.vosk_node.trigger_configure()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.vosk_node.destroy_node()
        rclpy.shutdown()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.tester_node = rclpy.create_node('tester_node')
        self.clock_pub = self.tester_node.create_publisher(Clock, '/clock', 1)
        self.audio_pub = self.tester_node.create_publisher(AudioData, '/audio/channel0', 1)
        self.vad_pub = self.tester_node.create_publisher(Bool, '/audio/voice_detected', 1)
        self.publishers_map = {
            '/audio/channel0': self.audio_pub,
            '/audio/voice_detected': self.vad_pub,
        }
        self.tester_executor = SingleThreadedExecutor()
        self.tester_executor.add_node(self.tester_node)

        self.hri_listener = HRIListener('hri_node', False)

        self.vosk_node.trigger_activate()
        return super().setUp()

    def tearDown(self) -> None:
        self.vosk_node.trigger_deactivate()
        del self.hri_listener
        self.tester_node.destroy_node()
        return super().tearDown()

    def spin(self, time_ns=None):
        if time_ns is not None:
            self.clock_pub.publish(Clock(clock=Time(nanoseconds=time_ns).to_msg()))
        spin_some(self.vosk_executor)
        self.hri_listener.spin_some(timedelta(seconds=1.))
        spin_some(self.tester_executor)

    def _test(self, bag_path: Path, expected_final: str):
        bag_reader = SequentialReader()
        print(str(bag_path))
        bag_reader.open(StorageOptions(uri=str(bag_path)), ConverterOptions('', ''))

        self.spin()
        self.assertEqual(len(self.hri_listener.voices), 0)

        while bag_reader.has_next():
            topic, msg_raw, time_ns = bag_reader.read_next()
            self.publishers_map[topic].publish(msg_raw)
            self.spin(time_ns)

        self.assertEqual(len(self.hri_listener.voices), 1)
        voice = next(iter(self.hri_listener.voices.values()))
        self.assertTrue(voice.id.startswith('voice_'))
        self.assertEquals(voice.incremental_speech, expected_final)
        self.assertEquals(voice.speech, expected_final)


class TestVoskEnglish(TestVoskMixin, unittest.TestCase):
    model = 'vosk_model_small'
    locale = 'en_US'
    bags_path = Path().cwd() / 'test' / 'data' / 'en_US'

    # def test_bag_0(self):
    #     self._test(self.bags_path / 'bag_0', 'bitch hit the robot')

    # def test_bag_1(self):
    #     self._test(self.bags_path / 'bag_1', 'i knew a girl but')

    def test_bag_2(self):
        self._test(self.bags_path / 'bag_2', 'hello')

    # def test_bag_3(self):
    #     self._test(self.bags_path / 'bag_3', 'hello how are you')

    def test_bag_4(self):
        self._test(self.bags_path / 'bag_4', 'what is your battery level')

    def test_bag_5(self):
        self._test(self.bags_path / 'bag_5', 'bye bye')

    # def test_bag_6(self):
    #     self._test(self.bags_path / 'bag_6', 'are you a boy')

    def test_bag_7(self):
        self._test(self.bags_path / 'bag_7', 'hi how are you')

    def test_bag_8(self):
        self._test(self.bags_path / 'bag_8', 'how old are you')

    # def test_bag_9(self):
    #     self._test(self.bags_path / 'bag_9', "you're ugly i hate you i hate you")

    # def test_bag_10(self):
    #     self._test(self.bags_path / 'bag_10', 'tell me a joke')

    def test_bag_11(self):
        self._test(self.bags_path / 'bag_11', 'where do you live')

    def test_bag_12(self):
        self._test(self.bags_path / 'bag_12', 'i love you')

    def test_bag_13(self):
        self._test(self.bags_path / 'bag_13', 'i love you')

    def test_bag_14(self):
        # actually two sentences, the first being "hello ari how are you"
        self._test(self.bags_path / 'bag_14', 'what is your name')

    # def test_bag_15(self):
    #     self._test(self.bags_path / 'bag_15', 'how is the weather')

    # def test_bag_16(self):
    #     self._test(self.bags_path / 'bag_16', 'what can you do')


if __name__ == '__main__':
    unittest.main()
