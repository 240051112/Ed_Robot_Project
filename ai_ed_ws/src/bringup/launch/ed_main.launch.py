from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch args
    model_path = DeclareLaunchArgument('model_path', default_value='/home/jetson/ultralytics/ultralytics/yolo11n.engine')
    confidence_threshold = DeclareLaunchArgument('confidence_threshold', default_value='0.5')
    show_video = DeclareLaunchArgument('show_video', default_value='true')

    # Static TF options (optional)
    publish_static_tf = DeclareLaunchArgument('publish_static_tf', default_value='false')
    base_frame  = DeclareLaunchArgument('base_frame',  default_value='base_link')
    camera_frame= DeclareLaunchArgument('camera_frame', default_value='camera_color_optical_frame')
    x = DeclareLaunchArgument('x', default_value='0.0')
    y = DeclareLaunchArgument('y', default_value='0.0')
    z = DeclareLaunchArgument('z', default_value='0.0')
    roll  = DeclareLaunchArgument('roll',  default_value='0.0')
    pitch = DeclareLaunchArgument('pitch', default_value='0.0')
    yaw   = DeclareLaunchArgument('yaw',   default_value='0.0')

    # Default params from ed_vision (installed file)
    ed_vision_share = get_package_share_directory('ed_vision')
    default_params = PathJoinSubstitution([ed_vision_share, 'config', 'ed_params.yaml'])

    # Perception node (ed_vision)
    perception = Node(
        package='ed_vision',
        executable='perception_node',
        name='ed_perception_node',
        output='screen',
        parameters=[
            default_params,
            {
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'show_video': LaunchConfiguration('show_video'),
            },
        ],
    )

    # Optional static TF (base -> camera)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ed_camera_static_tf',
        output='screen',
        arguments=[
            LaunchConfiguration('x'),
            LaunchConfiguration('y'),
            LaunchConfiguration('z'),
            LaunchConfiguration('roll'),
            LaunchConfiguration('pitch'),
            LaunchConfiguration('yaw'),
            LaunchConfiguration('base_frame'),
            LaunchConfiguration('camera_frame'),
        ],
        condition=IfCondition(LaunchConfiguration('publish_static_tf')),
    )

    return LaunchDescription([
        model_path, confidence_threshold, show_video,
        publish_static_tf, base_frame, camera_frame,
        x, y, z, roll, pitch, yaw,
        perception, static_tf,
    ])
