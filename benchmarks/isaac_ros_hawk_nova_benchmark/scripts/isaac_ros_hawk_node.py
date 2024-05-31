# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Live benchmarking Isaac ROS HawkNode.

The graph consists of the following:
- Graph under Test:
    1. CorrelatedTimestampDriver: timestamp correaltor
    2. HawkNode: publishes images

Required:
- Packages:
    - isaac_ros_correlated_timestamp_driver
    - isaac_ros_hawk
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BasicPerformanceCalculator, BenchmarkMode
from ros2_benchmark import ImageResolution
from ros2_benchmark import MonitorPerformanceCalculatorsInfo
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.HD


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS HawkNode."""
    correlated_timestamp_driver_node = ComposableNode(
        name='CorrelatedTimestampDriver',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='isaac_ros_correlated_timestamp_driver',
        plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
        parameters=[{'use_time_since_epoch': False,
                     'nvpps_dev_name': '/dev/nvpps0'}])

    hawk_node = ComposableNode(
        name='HawkNode',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{
            'module_id': 5,
        }]
    )

    left_image_monitor_node = ComposableNode(
        name='LeftImageMonitorNode',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
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
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 1,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'right/image_raw')]
    )

    left_camera_info_monitor_node = ComposableNode(
        name='LeftCameraInfoMonitorNode',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 2,
            'monitor_data_format': 'nitros_camera_info',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'left/camera_info')]
    )

    right_camera_info_monitor_node = ComposableNode(
        name='RightCameraInfoMonitorNode',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 3,
            'monitor_data_format': 'nitros_camera_info',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'right/camera_info')]
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSHawkNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            correlated_timestamp_driver_node,
            hawk_node,
            left_image_monitor_node,
            right_image_monitor_node,
            left_camera_info_monitor_node,
            right_camera_info_monitor_node,
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    return TestIsaacROSHawkNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSHawkNode(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS HawkNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS HawkNode Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION},
        monitor_info_list=[
            MonitorPerformanceCalculatorsInfo(
                'monitor_node0',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Left Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node1',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Right Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node2',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Left Camera Info',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node3',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Right Camera Info',
                    'message_key_match': True
                })])
        ]
    )

    def test_benchmark(self):
        self.run_benchmark()
