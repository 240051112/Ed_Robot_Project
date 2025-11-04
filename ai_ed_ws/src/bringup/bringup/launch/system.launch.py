from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='ed_skills',
             executable='echo_node',
             name='echo_node',
             output='screen')
    ])
