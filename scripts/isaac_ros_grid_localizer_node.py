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
Performance test for Isaac ROS OccupancyGridLocalizerNode.

- Preprocessors:
    1. PointCloudToFlatScanNode: converts point clouds to flat scans
- Graph under Test:
    1. OccupancyGridLocalizerNode: estimates poses relative to a map

Required:
- Packages:
    - isaac_ros_pointcloud_utils
    - isaac_ros_occupancy_grid_localizer
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
"""

import os

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_storage'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS OccupancyGridLocalizerNode."""

    MAP_YAML_PATH = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'occupancy_grid_localizer/maps/map.yaml')

    occupancy_grid_localizer_node = ComposableNode(
        name='OccupancyGridLocalizerNode',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='isaac_ros_occupancy_grid_localizer',
        plugin='nvidia::isaac_ros::occupancy_grid_localizer::OccupancyGridLocalizerNode',
        parameters=[
            MAP_YAML_PATH,
            {
                'loc_result_frame': 'map',
                'map_yaml_path': MAP_YAML_PATH,
            }
        ])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('pandar_xt_32_0_lidar', 'data_loader/pointcloud')]
    )

    pointcloud_to_flatscan_node = ComposableNode(
        name='PointCloudToFlatScanNode',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='isaac_ros_pointcloud_utils',
        plugin='nvidia::isaac_ros::pointcloud_utils::PointCloudToFlatScanNode',
        remappings=[('pointcloud', 'data_loader/pointcloud'),
                    ('flatscan', 'buffer/flatscan_localization')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_flat_scan']
        }],
        remappings=[('buffer/input0', 'buffer/flatscan_localization'),
                    ('input0', 'flatscan_localization')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_pose_cov_stamped',
            'use_nitros_type_monitor_sub': True
        }],
        remappings=[
            ('output', 'localization_result')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSOccupancyGridLocalizerNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            pointcloud_to_flatscan_node,
            playback_node,
            monitor_node,
            occupancy_grid_localizer_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    return TestIsaacROSOccupancyGridLocalizerNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSOccupancyGridLocalizerNode(ROS2BenchmarkTest):
    """Performance test for Isaac ROS OccupancyGridLocalizerNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Occupancy Grid Localizer Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=50.0,
        publisher_lower_frequency=1.0,
        # The number of frames to be buffered
        playback_message_buffer_size=5,
        # Frequency, in hz, to increase the target frequency by with each step
        linear_scan_step_size=1.0,
        # Frame rate drop between input and output that can be tolerated without failing the test
        binary_search_acceptable_frame_rate_drop=1,
        # Adding offset to prevent using first pointcloud msg, which is partially accumulated
        # Input Data Start Time (s)
        input_data_start_time=1
    )

    def test_benchmark(self):
        self.run_benchmark()
