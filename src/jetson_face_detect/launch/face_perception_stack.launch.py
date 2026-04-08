from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    jetson_face_detect_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{get_package_share_directory('jetson_face_detect')}/launch/ros4hri_face_node.launch.py"
        )
    )

    face_identification_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{get_package_share_directory('hri_face_identification')}/launch/"
            "hri_face_identification.launch.py"
        )
    )

    engagement_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f"{get_package_share_directory('hri_engagement')}/launch/hri_engagement.launch.py"
        )
    )

    return LaunchDescription([
        jetson_face_detect_launch,
        face_identification_launch,
        engagement_launch,
    ])
