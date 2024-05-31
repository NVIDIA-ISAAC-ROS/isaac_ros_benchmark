# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Performance test for Isaac ROS Cumotion node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BenchmarkMode
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

# 20 Hz planning rate
TESTING_SLEEP_TIME = 0.05
# Joint names for franka robot
JOINT_NAMES = ['panda_finger_joint1', 'panda_joint1', 'panda_joint2',
               'panda_finger_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
               'panda_joint6', 'panda_joint7']
# Joint start and end position were estimated by
# 1. Running the cumotion planner node + Franak model + rviz tutorial
# 2. Move the end effector to a random position+orienation
# 3. Check if it is reachable by clicking the "plan" button
# 4. If it is reachable, then open a new terminal and run "ros2 topic echo /joint_states"
# 5. The current value of the joints are the start position
# 6. Click on "plan and execute"
# 7. The value after the robot finishes moving are the end positions
JOINT_START_POSITIONS = [0.0, 0.0, -0.785, 0.0, 0.0, -2.356, 0.0, 1.571, 0.785]
JOINT_END_POSITIONS = [0.0, 0.0, -0.29027499956392067, 0.0,
                       0.0, -1.1287, 0.0,
                       0.8385, 0.785]


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS Cumotion node."""
    isaac_ros_cumotion_benchmark_dir = get_package_share_directory('isaac_ros_cumotion_benchmark')
    cumotion_planner_node = Node(
            package='isaac_ros_cumotion',
            executable='cumotion_planner_node',
            name='cumotion_planner_node',
            namespace=TestIsaacROSCumotionGraph.generate_namespace(),
            output='screen',
            parameters=[
                {'robot': 'franka.xrdf'},
                {'urdf_path': os.path.join(isaac_ros_cumotion_benchmark_dir, 'urdf', 'panda.urdf')}
            ]
    )

    cumotion_action_client_node = Node(
            package='isaac_ros_cumotion_benchmark',
            executable='cumotion_action_client.py',
            name='cumotion_action_client',
            namespace=TestIsaacROSCumotionGraph.generate_namespace(),
            output='screen',
            parameters=[
                {'joint_names': JOINT_NAMES},
                {'joint_start_positions': JOINT_START_POSITIONS},
                {'joint_end_positions': JOINT_END_POSITIONS},
                {'sleep_time': TESTING_SLEEP_TIME}
            ]
        )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSCumotionGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::MonitorNode',
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'geometry_msgs/msg/PoseStamped',
        }],
        remappings=[
            ('output', 'cumotion_planning_success')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSCumotionGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            monitor_node,
        ],
        output='screen'
    )

    return [composable_node_container, cumotion_planner_node, cumotion_action_client_node]


def generate_test_description():
    return TestIsaacROSCumotionGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSCumotionGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Cumotion node."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Cumotion Node Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        pre_trial_run_wait_time_sec=60.0,
    )

    def test_benchmark(self):
        self.run_benchmark()
