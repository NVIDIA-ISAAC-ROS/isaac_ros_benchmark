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
Performance test for the Isaac ROS ESS depth (1 throttled light ESS) graph at 30Hz.

Required:
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hideaway
- Models:
    - assets/models/ess/light_ess.onnx
"""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_benchmark import NitrosMonitorUtility

import isaac_ros_ess_benchmark.ess_benchmark_utility as ess_benchmark_utility
import isaac_ros_ess_benchmark.ess_model_utility as ess_model_utility

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hideaway'

NITROS_MONITOR_UTILITY = NitrosMonitorUtility()


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS ESS depth graph."""
    asset_models_path = os.path.join(TestIsaacROSEssDepthGraph.get_assets_root_path(), 'models')
    _, light_ess_engine_file_path = ess_model_utility.get_model_paths('light', asset_models_path)
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

    launch_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_ess_benchmark'), 'scripts', 'include')

    light_ess_depth_include_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([launch_include_dir, '/ess_depth.include.py']),
        launch_arguments={
            'container_name': 'benchmark_container',
            'node_namespace': TestIsaacROSEssDepthGraph.generate_namespace(),
            'ess_model_type': 'light',
            'engine_file_path': light_ess_engine_file_path,
            'ess_throttler_skip': '1',
        }.items(),
    )
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSEssDepthGraph.generate_namespace(),
        message_key_match=False))

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSEssDepthGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('hawk_0_left_rgb_image', 'data_loader/left_image'),
            ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
            ('hawk_0_right_rgb_image', 'data_loader/right_image'),
            ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')
        ]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSEssDepthGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_image_rgb8',
                'nitros_camera_info',
                'nitros_image_rgb8',
                'nitros_camera_info'
            ],
        }],
        remappings=[
            ('buffer/input0', 'data_loader/left_image'),
            ('input0', 'left/image_raw'),
            ('buffer/input1', 'data_loader/left_camera_info'),
            ('input1', 'left/camera_info'),
            ('buffer/input2', 'data_loader/right_image'),
            ('input2', 'right/image_raw'),
            ('buffer/input3', 'data_loader/right_camera_info'),
            ('input3', 'right/camera_info')
        ]
    )

    load_benchmark_nodes = LoadComposableNodes(
        target_container='benchmark_container',
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
        ] + monitor_nodes
    )

    return [benchmark_container, light_ess_depth_include_1, load_benchmark_nodes]


def generate_test_description():
    asset_models_path = os.path.join(TestIsaacROSEssDepthGraph.get_assets_root_path(), 'models')
    ess_model_utility.generate_ess_engine_file('light', asset_models_path)
    return TestIsaacROSEssDepthGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSEssDepthGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS ESS depth graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS ESS Depth (1 Throttled Light ESS) Graph 30Hz Benchmark',
        input_data_path=ROSBAG_PATH,
        # Benchmark at 30Hz
        publisher_upper_frequency=30.0,
        publisher_lower_frequency=30.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10,
        pre_trial_run_wait_time_sec=5.0,
        additional_fixed_publisher_rate_tests=[],
        monitor_info_list=[]
    )

    def test_benchmark(self):
        self.config.monitor_info_list = NITROS_MONITOR_UTILITY.get_monitor_info_list()
        self.run_benchmark()
