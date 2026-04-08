#!/usr/bin/env python3

import signal
import subprocess
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioStamped


class AlsaMicPublisher(Node):
    def __init__(self):
        super().__init__('alsa_mic_publisher')

        # ALSA capture settings
        self.declare_parameter('alsa_device', 'hw:1,0')
        self.declare_parameter('input_rate', 48000)
        self.declare_parameter('input_channels', 1)
        self.declare_parameter('input_chunk', 1024)

        # Published output settings
        self.declare_parameter('output_rate', 16000)
        self.declare_parameter('output_chunk', 512)

        # ROS / behavior settings
        self.declare_parameter('topic_name', '/audio')
        self.declare_parameter('frame_id', 'usb_audio_mic')
        self.declare_parameter('noise_gate_rms', 0.0)
        self.declare_parameter('log_every_n', 0)

        self.alsa_device = str(self.get_parameter('alsa_device').value)
        self.input_rate = int(self.get_parameter('input_rate').value)
        self.input_channels = int(self.get_parameter('input_channels').value)
        self.input_chunk = int(self.get_parameter('input_chunk').value)

        self.output_rate = int(self.get_parameter('output_rate').value)
        self.output_chunk = int(self.get_parameter('output_chunk').value)

        self.topic_name = str(self.get_parameter('topic_name').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.noise_gate_rms = float(self.get_parameter('noise_gate_rms').value)
        self.log_every_n = int(self.get_parameter('log_every_n').value)

        if self.input_channels != 1:
            raise ValueError('This script supports mono ALSA capture only.')

        if self.input_rate != 48000 or self.output_rate != 16000:
            self.get_logger().warning(
                f'This script is intended for 48000 -> 16000. Current: {self.input_rate} -> {self.output_rate}'
            )

        if self.input_rate % self.output_rate != 0:
            raise ValueError(
                f'Expected integer downsample ratio, got {self.input_rate} -> {self.output_rate}'
            )

        self.downsample_factor = self.input_rate // self.output_rate
        if self.downsample_factor != 3:
            self.get_logger().warning(
                f'Expected downsample factor 3 for 48k -> 16k, got {self.downsample_factor}'
            )

        self.publisher_ = self.create_publisher(AudioStamped, self.topic_name, 10)

        self.proc = None
        self.timer = None
        self._pub_count = 0
        self._shutting_down = False

        # Buffer of 16 kHz samples so we can publish exact 512-sample frames
        self.output_buffer = np.array([], dtype=np.int16)

        self.start_arecord()

        timer_period = self.input_chunk / float(self.input_rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Capturing from {self.alsa_device} at {self.input_rate} Hz, '
            f'publishing {self.topic_name} at {self.output_rate} Hz with chunk={self.output_chunk}'
        )

    def start_arecord(self):
        # -t raw writes raw PCM to stdout
        # -q keeps arecord quieter
        cmd = [
            'arecord',
            '-D', self.alsa_device,
            '-q',
            '-f', 'S16_LE',
            '-c', str(self.input_channels),
            '-r', str(self.input_rate),
            '-t', 'raw'
        ]

        self.get_logger().info(f'Starting ALSA capture: {" ".join(cmd)}')

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

        if self.proc.stdout is None:
            raise RuntimeError('Failed to open arecord stdout pipe.')

    def timer_callback(self):
        if self._shutting_down or self.proc is None or self.proc.stdout is None:
            return

        bytes_per_sample = 2  # S16_LE
        bytes_needed = self.input_chunk * self.input_channels * bytes_per_sample

        try:
            raw_bytes = self.proc.stdout.read(bytes_needed)
        except Exception as e:
            if not self._shutting_down:
                self.get_logger().error(f'ALSA read failed: {e}')
            return

        if not raw_bytes:
            if self.proc.poll() is not None and not self._shutting_down:
                stderr_text = ''
                try:
                    if self.proc.stderr is not None:
                        stderr_text = self.proc.stderr.read().decode(errors='ignore').strip()
                except Exception:
                    pass
                self.get_logger().error(
                    f'arecord exited with code {self.proc.returncode}. stderr: {stderr_text}'
                )
            return

        if len(raw_bytes) < bytes_needed:
            # Partial read; skip this round
            return

        samples_48k = np.frombuffer(raw_bytes, dtype=np.int16)
        if samples_48k.size == 0:
            return

        # 48k -> 16k by decimation
        samples_16k = samples_48k[::self.downsample_factor].astype(np.int16)
        if samples_16k.size == 0:
            return

        self.output_buffer = np.concatenate((self.output_buffer, samples_16k))

        while self.output_buffer.size >= self.output_chunk:
            frame = self.output_buffer[:self.output_chunk]
            self.output_buffer = self.output_buffer[self.output_chunk:]

            if self.noise_gate_rms > 0.0:
                frame_float = frame.astype(np.float32)
                rms = float(np.sqrt(np.mean(frame_float * frame_float)))
                if rms < self.noise_gate_rms:
                    continue

            self._pub_count += 1
            if self.log_every_n > 0 and self._pub_count % self.log_every_n == 0:
                frame_float = frame.astype(np.float32)
                rms = float(np.sqrt(np.mean(frame_float * frame_float)))
                peak = int(np.max(np.abs(frame)))
                self.get_logger().info(
                    f'published={self._pub_count} RMS={rms:.1f} PEAK={peak}'
                )

            msg = AudioStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.audio.audio_data.int16_data = frame.tolist()
            msg.audio.info.format = 8  # paInt16 / 16-bit PCM
            msg.audio.info.channels = 1
            msg.audio.info.rate = self.output_rate
            msg.audio.info.chunk = self.output_chunk

            self.publisher_.publish(msg)

    def close_capture(self):
        self._shutting_down = True

        try:
            if self.timer is not None:
                self.destroy_timer(self.timer)
                self.timer = None
        except Exception:
            pass

        try:
            if self.proc is not None:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
        except Exception:
            try:
                if self.proc is not None:
                    self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None

    def destroy_node(self):
        self.close_capture()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    def handle_signal(signum, frame):
        nonlocal node
        if node is not None:
            try:
                node.close_capture()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        node = AlsaMicPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Fatal error: {e}', file=sys.stderr)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
