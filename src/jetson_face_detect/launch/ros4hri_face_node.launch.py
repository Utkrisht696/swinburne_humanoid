from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    performance_debug = LaunchConfiguration('performance_debug')
    use_cuda_preprocess = LaunchConfiguration('use_cuda_preprocess')
    use_hw_jpeg_decode = LaunchConfiguration('use_hw_jpeg_decode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'performance_debug',
            default_value='false',
            description='Enable periodic detector performance logging',
        ),
        DeclareLaunchArgument(
            'use_cuda_preprocess',
            default_value='true',
            description='Use CUDA resize/color preprocessing when available',
        ),
        DeclareLaunchArgument(
            'use_hw_jpeg_decode',
            default_value='true',
            description='Use Jetson GStreamer nvjpegdec path for CompressedImage decode when available',
        ),
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
                    'conf_threshold': 0.65,
                    'nms_threshold': 0.4,
                    'min_face_size': 80,
                    'tracker_iou_threshold': 0.35,
                    'tracker_min_predicted_iou': 0.05,
                    'tracker_max_center_distance_ratio': 0.9,
                    'tracker_max_missed': 15,
                    'tracker_debug': False,
                    'performance_debug': performance_debug,
                    'publish_debug_image': True,
                    'publish_crops': True,
                    'publish_aligned': True,
                    'apply_sigmoid': False,
                    'detector_debug': False,
                    'detector_max_rate_hz': 12.0,
                    'redetect_on_missed': 1,
                    'debug_publish_rate_hz': 10.0,
                    'publish_face_assets_on_detect_only': True,
                    'use_cuda_preprocess': use_cuda_preprocess,
                    'use_hw_jpeg_decode': use_hw_jpeg_decode,
                    'filter_non_frontal_faces': True,
                    'frontal_max_nose_offset_ratio': 1.50,
                    'frontal_max_mouth_offset_ratio': 1.50,
                    'frontal_max_eye_y_diff_ratio': 1.50,
                    'frontal_min_eye_mouth_ratio': 0.10,
                    'frontal_max_eye_mouth_ratio': 5.0,
                    'filter_low_quality_identification_faces': True,
                    'identification_require_detector_confirmation': True,
                    'id_quality_min_face_stddev': 18.0,
                    'id_quality_min_patch_stddev': 10.0,
                    'id_quality_min_patch_laplacian_var': 45.0,
                    'id_quality_min_pass_ratio': 0.75,
                }
            ]
        )
    ])
