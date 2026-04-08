from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='jetson_face_detect',
            executable='ros4hri_face_node',
            name='ros4hri_face_node',
            output='screen',
            parameters=[
                {
                    'image_topic': '/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed',
                    'camera_info_topic': '/aima/hal/sensor/rgb_head_front_center/camera_info',
                    'frame_id': 'rgb_head_center',
                    'engine_path': '/home/agi/models/det_2.5g_fp16.engine',
                    'debug_width': 450,
                    'jpeg_quality': 50,
                    'publish_landmarks': True,
                    'crop_size': 128,
                    'max_faces': 3,
                    'conf_threshold': 0.55,
                    'nms_threshold': 0.4,
                    'min_face_size': 80,
                    'tracker_iou_threshold': 0.35,
                    'tracker_min_predicted_iou': 0.05,
                    'tracker_max_center_distance_ratio': 0.9,
                    'tracker_max_missed': 6,
                    'tracker_debug': False,
                    'performance_debug': False,
                    'publish_debug_image': True,
                    'publish_crops': True,
                    'publish_aligned': True,
                    'apply_sigmoid': False,
                    'detector_debug': False,
                }
            ]
        )
    ])
