import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def _robot_description():
    urdf_path = os.path.join(
        get_package_share_directory("x2_description"),
        "urdf",
        "x2_ultra.urdf",
    )

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        return urdf_file.read()


def generate_launch_description():
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    _robot_description(),
                    value_type=str,
                )
            }
        ],
    )

    base_footprint_projector_node = Node(
        package="x2_description",
        executable="base_footprint_projector.py",
        name="base_footprint_projector",
        output="screen",
        parameters=[
            {
                "base_frame": "base_link",
                "footprint_frame": "base_footprint",
                "ground_contact_frames": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ],
                "default_footprint_height": 0.67195,
                "publish_rate": 50.0,
            }
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            base_footprint_projector_node,
        ]
    )
