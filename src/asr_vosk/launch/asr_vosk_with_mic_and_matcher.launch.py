from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    alsa_device = LaunchConfiguration('alsa_device')
    model = LaunchConfiguration('model')
    default_locale = LaunchConfiguration('default_locale')
    use_vosk_venv = LaunchConfiguration('use_vosk_venv')
    asr_executable = PythonExpression(
        ['"pal_asr_vosk" if ', use_vosk_venv, ' == True else "asr_vosk"']
    )

    asr_node = LifecycleNode(
        package='asr_vosk',
        executable=asr_executable,
        namespace='',
        name='asr_vosk',
        output='screen',
        parameters=[{
            'model': model,
            'default_locale': default_locale,
        }],
    )

    configure_event = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(asr_node),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))

    activate_event = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=asr_node,
        goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(asr_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ))],
        handle_once=True,
    ))

    return LaunchDescription([
        DeclareLaunchArgument('alsa_device', default_value='hw:1,0'),
        DeclareLaunchArgument('model', default_value='vosk_model_small'),
        DeclareLaunchArgument('default_locale', default_value='en_US'),
        DeclareLaunchArgument('use_vosk_venv', default_value='False'),
        Node(
            package='asr_vosk',
            executable='alsa_mic',
            name='alsa_mic_publisher',
            output='screen',
            parameters=[{
                'alsa_device': alsa_device,
                'topic_name': '/audio',
                'output_rate': 16000,
                'output_chunk': 512,
            }],
        ),
        asr_node,
        configure_event,
        activate_event,
        Node(
            package='hri_voice_face_matcher',
            executable='hri_voice_face_matcher',
            name='hri_voice_face_matcher',
            output='screen',
        ),
    ])
