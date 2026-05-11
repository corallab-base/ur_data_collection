from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    GroupAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    NotSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, PushROSNamespace
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_rviz_arg = DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")

    robot_launch_file = PathJoinSubstitution(
        [FindPackageShare("ur_data_collection"), "launch", "robot.launch.py"]
    )

    launch_rviz = LaunchConfiguration("launch_rviz")

    robot_launch =IncludeLaunchDescription(
        robot_launch_file,
        launch_arguments={
            "robot_ip": "10.168.4.249",
            "ur_type": "ur5e",
            "launch_rviz": launch_rviz,
            # "initial_joint_controller": "freedrive_mode_controller",
            "reverse_port": "50005",
            "script_sender_port": "50006",
            "trajectory_port": "50007",
            "script_command_port": "50008",
        }.items(),
    )

    # free drive activator

    enable_freedrive = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--rate", "2",
            "/freedrive_mode_controller/enable_freedrive_mode",
            "std_msgs/msg/Bool", "{data: true}",
        ],
    )

    # camera

    rs_launch_file = PathJoinSubstitution(
        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
    )

    realsense = IncludeLaunchDescription(
        rs_launch_file,
        launch_arguments={
            "camera_name": "camera",
            "enable_rgbd": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "enable_color": "true",
            "enable_depth": "true",
            "rgb_camera.color_profile": "640x360x15",
            "rgb_camera.enable_auto_exposure": "false",
            "rgb_camera.exposure": "600",
            "depth_module.depth_profile": "640x360x15",
        }.items(),
    )

    # samurai

    # samurai = Node(
    #     package="coral_trackers",
    #     executable="samurai_tracker",
    #     parameters=[
    #         {"rgb_topic": "/camera/camera/color/image_raw"},
    #         {"samurai_config": "configs/samurai/sam2.1_hiera_t.yaml"},
    #         {"samurai_checkpoint": "/home/tassos/phd/software/samurai/sam2/checkpoints/sam2.1_hiera_tiny.pt"},
    #     ],
    # )

    return LaunchDescription([
        launch_rviz_arg,
        robot_launch,
        enable_freedrive,
        realsense,
        # samurai,
    ])
