from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='arm',
            executable='arm_node',
            name='arm_vision_task',
            output='screen',
        ),
        Node(
            package='arithmetic_problem',
            executable='arithmetic_problem',
            name='arithmetic_problem_node',
            output='screen',
        ),
    ])
