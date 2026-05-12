from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushROSNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    episode_path_arg = DeclareLaunchArgument(
        "episode_path", default_value="",
        description="Absolute path to the episode .pkl file to replay",
    )
    rate_hz_arg = DeclareLaunchArgument(
        "rate_hz", default_value="10.0",
        description="Playback rate in Hz",
    )
    loop_arg = DeclareLaunchArgument(
        "loop", default_value="true",
        description="Loop the episode continuously",
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz", default_value="true",
        description="Launch RViz?",
    )
    ur_type_arg = DeclareLaunchArgument(
        "ur_type", default_value="ur5e",
        description="UR robot type for URDF generation",
    )

    kinematics_params = PathJoinSubstitution(
        [FindPackageShare("ur_data_collection"), "config", "kinematics_params.yaml"]
    )

    # robot_state_publisher via ur_rsp — use_fake_hardware=true so no real robot is needed
    rsp = GroupAction(actions=[
        PushROSNamespace("fake"),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution(
                    [FindPackageShare("ur_robot_driver"), "launch", "ur_rsp.launch.py"]
                )
            ),
            launch_arguments={
                "robot_ip": "10.168.4.249",
                "ur_type": LaunchConfiguration("ur_type"),
                "use_fake_hardware": "true",
                "kinematics_params_file": kinematics_params,
                "tf_prefix": "replay_",
            }.items(),
        )
    ])

    replay_node = Node(
        package="ur_data_collection",
        executable="replay_episode",
        parameters=[{
            "episode_path": LaunchConfiguration("episode_path"),
            "rate_hz": LaunchConfiguration("rate_hz"),
            "loop": LaunchConfiguration("loop"),
            "tf_prefix": "replay_",
        }],
        output="screen",
    )

    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ur_data_collection"), "config", "replay_episode.rviz"]
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
        output="screen",
    )

    return LaunchDescription([
        episode_path_arg,
        rate_hz_arg,
        loop_arg,
        launch_rviz_arg,
        ur_type_arg,
        rsp,
        replay_node,
        rviz,
    ])
