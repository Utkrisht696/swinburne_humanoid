from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    body_detect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{get_package_share_directory('hri_body_detect')}/launch/hri_body_detect.launch.py"
        )
    )

    face_body_matcher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{get_package_share_directory('hri_face_body_matcher')}/launch/"
            "hri_face_body_matcher.launch.py"
        )
    )

    return LaunchDescription([
        body_detect_launch,
        face_body_matcher_launch,
    ])
