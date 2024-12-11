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
"""
Live performance test for the Isaac ROS Hawk ESS depth (1 full 3 throttled light ESS) graph.

Required:
- Models:
    - assets/models/ess/ess.onnx
    - assets/models/ess/light_ess.onnx
"""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_benchmark import NitrosMonitorUtility

import isaac_ros_ess_benchmark.ess_benchmark_utility as ess_benchmark_utility
import isaac_ros_ess_benchmark.ess_model_utility as ess_model_utility
import isaac_ros_hawk_nova_benchmark.hawk_benchmark_utility as hawk_benchmark_utility

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes, Node

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest

TYPE_NEGOTIATION_DURATION_S = '10'
NITROS_MONITOR_UTILITY = NitrosMonitorUtility()


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS Hawk ESS depth graph."""
    asset_models_path = os.path.join(
        TestIsaacROSHawkEssDepthGraph.get_assets_root_path(), 'models')
    _, full_ess_engine_file_path = ess_model_utility.get_model_paths('full', asset_models_path)
    _, light_ess_engine_file_path = ess_model_utility.get_model_paths('light', asset_models_path)
    result_load_list = []
    monitor_nodes = []
    benchmark_container = Node(
        name='benchmark_container',
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        output='screen',
        arguments=[
            '--ros-args', '--log-level', 'info',
        ]
    )

    hawk_launch_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_hawk_nova_benchmark'), 'scripts', 'include')
    ess_launch_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_ess_benchmark'), 'scripts', 'include')

    # Front Hawk
    node_namespace = 'front_full_ess'
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([hawk_launch_include_dir, '/hawk.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'hawk_placement': 'front',
            'create_correlated_timestamp_driver_node': 'True',
        }.items(),
    ))
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace))

    # Front Full ESS
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ess_launch_include_dir, '/ess_depth.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'ess_model_type': 'full',
            'engine_file_path': full_ess_engine_file_path,
        }.items(),
    ))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace,
        message_key_match=True))

    # Left Hawk
    node_namespace = 'left_light_ess'
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([hawk_launch_include_dir, '/hawk.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'hawk_placement': 'left',
            'create_correlated_timestamp_driver_node': 'False',
        }.items(),
    ))
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace))

    # Left Throttled Light ESS
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ess_launch_include_dir, '/ess_depth.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'ess_model_type': 'light',
            'engine_file_path': light_ess_engine_file_path,
            'ess_throttler_skip': '1',
        }.items(),
    ))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace,
        message_key_match=True))

    # Right Hawk
    node_namespace = 'right_light_ess'
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([hawk_launch_include_dir, '/hawk.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'hawk_placement': 'right',
            'create_correlated_timestamp_driver_node': 'False',
        }.items(),
    ))
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace))

    # Right Throttled Light ESS
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ess_launch_include_dir, '/ess_depth.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'ess_model_type': 'light',
            'engine_file_path': light_ess_engine_file_path,
            'ess_throttler_skip': '1',
        }.items(),
    ))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace,
        message_key_match=True))

    # Back Hawk
    node_namespace = 'back_light_ess'
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([hawk_launch_include_dir, '/hawk.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'hawk_placement': 'back',
            'create_correlated_timestamp_driver_node': 'False',
        }.items(),
    ))
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace))

    # Back Throttled Light ESS
    result_load_list.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ess_launch_include_dir, '/ess_depth.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': node_namespace,
            'type_negotiation_duration_s': TYPE_NEGOTIATION_DURATION_S,
            'ess_model_type': 'light',
            'engine_file_path': light_ess_engine_file_path,
            'ess_throttler_skip': '1',
        }.items(),
    ))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSHawkEssDepthGraph.generate_namespace(),
        node_namespace,
        message_key_match=True))

    load_benchmark_nodes = LoadComposableNodes(
        target_container='benchmark_container',
        composable_node_descriptions=monitor_nodes
    )

    return [benchmark_container, load_benchmark_nodes] + result_load_list


def generate_test_description():
    asset_models_path = os.path.join(
        TestIsaacROSHawkEssDepthGraph.get_assets_root_path(), 'models')
    # Generate engine files
    # Full ESS model
    ess_model_utility.generate_ess_engine_file('full', asset_models_path)
    # Light ESS model
    ess_model_utility.generate_ess_engine_file('light', asset_models_path)
    return TestIsaacROSHawkEssDepthGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSHawkEssDepthGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Hawk ESS depth graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Hawk ESS Depth (1 Full 3 Throttled Light ESS) '
                       'Graph Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        pre_trial_run_wait_time_sec=20.0,
        monitor_info_list=[]
    )

    def test_benchmark(self):
        self.config.monitor_info_list = NITROS_MONITOR_UTILITY.get_monitor_info_list()
        self.run_benchmark()
