# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Live benchmarking Isaac ROS ArgusStereoNode.

The graph consists of the following:
- Graph under Test:
    1. ArgusStereoNode: publishes images

Required:
- Packages:
    - isaac_ros_argus_camera
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import BasicPerformanceCalculator, BenchmarkMode
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest
from ros2_benchmark import MonitorPerformanceCalculatorsInfo

IMAGE_RESOLUTION = ImageResolution.HD

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS ArgusStereoNode."""
    argus_node = ComposableNode(
        name='ArgusStereoNode',
        namespace=TestIsaacROSArgusStereoNode.generate_namespace(),
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{
            'module_id': 5,
        }]
    )

    left_image_monitor_node = ComposableNode(
        name='LeftImageMonitorNode',
        namespace=TestIsaacROSArgusStereoNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'left/image_raw')]
    )

    right_image_monitor_node = ComposableNode(
        name='RightImageMonitorNode',
        namespace=TestIsaacROSArgusStereoNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 1,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'right/image_raw')]
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSArgusStereoNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            argus_node,
            left_image_monitor_node,
            right_image_monitor_node,
        ],
        output='screen'
    )

    return [composable_node_container]

def generate_test_description():
    return TestIsaacROSArgusStereoNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSArgusStereoNode(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS ArgusStereoNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS ArgusStereoNode Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION},
        monitor_info_list=[
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring0',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Argus Left Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring1',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Argus Right Image',
                    'message_key_match': True
                })])
        ]
    )

    def test_benchmark(self):
        self.run_benchmark()
