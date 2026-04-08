from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hri_voice_face_matcher',
            executable='hri_voice_face_matcher',
            name='hri_voice_face_matcher',
            output='screen',
        )
    ])
