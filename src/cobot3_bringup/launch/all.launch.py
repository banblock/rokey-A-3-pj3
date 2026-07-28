"""Start all ROS 2 nodes used by the shoe recycling system."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_control = LaunchConfiguration("use_control")
    use_vision = LaunchConfiguration("use_vision")
    use_fms = LaunchConfiguration("use_fms")
    use_ui = LaunchConfiguration("use_ui")
    use_main_control_stub = LaunchConfiguration("use_main_control_stub")
    config_module = LaunchConfiguration("config_module")
    save_debug_image = LaunchConfiguration("save_debug_image")

    vision_resource = PathJoinSubstitution(
        [FindPackageShare("vision_node"), "resource"]
    )

    control_node = Node(
        package="recycle_controller",
        executable="control",
        name="control_node",
        output="screen",
        condition=IfCondition(use_control),
    )

    vision_node = Node(
        package="vision_node",
        executable="vision_node",
        name="vision_node",
        output="screen",
        condition=IfCondition(use_vision),
        parameters=[{
            "model_path": PathJoinSubstitution(
                [vision_resource, "best_task1.pt"]
            ),
            "model_path_stage2": PathJoinSubstitution(
                [vision_resource, "best_task2.pt"]
            ),
            "model_path_stage3": PathJoinSubstitution(
                [vision_resource, "best_task3.pt"]
            ),
            "save_debug_image": save_debug_image,
        }],
    )

    fms_node = Node(
        package="sorting_line_fms",
        executable="fms_node",
        name="fleet_management_system",
        output="screen",
        condition=IfCondition(use_fms),
    )

    fleet_driver = Node(
        package="sorting_line_fms",
        executable="fleet_driver",
        name="fleet_driver",
        output="screen",
        condition=IfCondition(use_fms),
        parameters=[{"config_module": config_module}],
    )

    main_control_stub = Node(
        package="sorting_line_fms",
        executable="main_control_stub",
        name="main_control_stub",
        output="screen",
        condition=IfCondition(use_main_control_stub),
    )

    ui_node = Node(
        package="ui_node",
        executable="ui",
        name="dashboard_ui",
        output="screen",
        condition=IfCondition(use_ui),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_control", default_value="true"),
        DeclareLaunchArgument("use_vision", default_value="true"),
        DeclareLaunchArgument("use_fms", default_value="true"),
        DeclareLaunchArgument("use_ui", default_value="true"),
        DeclareLaunchArgument(
            "use_main_control_stub",
            default_value="false",
            description=(
                "Publish test pickup jobs. Keep false when recycle_controller "
                "is running."
            ),
        ),
        DeclareLaunchArgument(
            "config_module",
            default_value="fleet_config_test1",
            description="Graph configuration shared by the Fleet Driver.",
        ),
        DeclareLaunchArgument(
            "save_debug_image",
            default_value="false",
            description="Save Vision debug images to disk.",
        ),
        control_node,
        vision_node,
        fms_node,
        # Give FMS time to create its subscriptions before driver registration.
        TimerAction(period=2.0, actions=[fleet_driver, main_control_stub]),
        # Start the GUI after the service providers have had time to initialize.
        TimerAction(period=3.0, actions=[ui_node]),
    ])
