#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    TransformBroadcaster,
    TransformException,
    TransformListener,
)


class BaseFootprintProjector(Node):
    def __init__(self):
        super().__init__("base_footprint_projector")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("footprint_frame", "base_footprint")
        self.declare_parameter(
            "ground_contact_frames",
            ["left_ankle_roll_link", "right_ankle_roll_link"],
        )
        self.declare_parameter("default_footprint_height", 0.67195)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("transform_timeout", 0.02)

        self.base_frame = self.get_parameter("base_frame").value
        self.footprint_frame = self.get_parameter("footprint_frame").value
        self.ground_contact_frames = list(
            self.get_parameter("ground_contact_frames").value
        )
        self.default_footprint_height = float(
            self.get_parameter("default_footprint_height").value
        )
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.transform_timeout = Duration(
            seconds=float(self.get_parameter("transform_timeout").value)
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_height = self.default_footprint_height

        timer_period = 1.0 / publish_rate if publish_rate > 0.0 else 0.02
        self.timer = self.create_timer(timer_period, self.publish_footprint)

    def publish_footprint(self):
        height = self._current_base_height()
        if height is None:
            height = self.last_height
        else:
            self.last_height = height

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.footprint_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = height
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(transform)

    def _current_base_height(self):
        ground_z_values = []

        for frame in self.ground_contact_frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    frame,
                    Time(),
                    timeout=self.transform_timeout,
                )
            except TransformException as exc:
                self.get_logger().debug(
                    f"Could not transform {frame} to {self.base_frame}: {exc}"
                )
                continue

            ground_z_values.append(transform.transform.translation.z)

        if not ground_z_values:
            return None

        return max(0.0, -min(ground_z_values))


def main(args=None):
    rclpy.init(args=args)
    node = BaseFootprintProjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
