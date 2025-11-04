"""
system.launch.py  –  boots Ed’s brain + Yahboom arm driver
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # ── Ed’s dialogue / RAG brain ────────────────────────────
    ld.add_action(
        Node(
            package="ed_core",
            executable="ed_brain",
            name="ed_brain",
            output="screen",
        )
    )

    # ── Low-level Yahboom arm driver ─────────────────────────
    ld.add_action(
        Node(
            package="ed_skills",
            executable="arm_driver",      # comes from setup.cfg entry-point
            name="arm_driver",
            output="screen",
        )
    )

    # ── (optional) voice interface – enable later ────────────
    # ld.add_action(
    #     Node(
    #         package="ed_io_voice",
    #         executable="ed_voice_node",
    #         name="ed_voice",
    #         parameters=[{"model": "vosk-model-small-en-us-0.15"}],
    #         output="screen",
    #     )
    # )

    return ld
