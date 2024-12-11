# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for the Isaac AprilTag graph.

The graph consists of the following:
- Preprocessors:
    1. PrepResizeNode: resizes images to HD
- Graph under Test:
    1. RectifyNode: rectifies images
    2. AprilTagNode: detects Apriltags

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_apriltag
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.HD
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_storage'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS AprilTag graph."""
    rectify_node = ComposableNode(
        name='RectifyNode',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }]
    )

    apriltag_node = ComposableNode(
        name='AprilTagNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='isaac_ros_apriltag',
        plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
        remappings=[
            ('image', 'image_rect'),
            ('camera_info', 'camera_info_rect'),
            ('tag_detections', 'apriltag_detections')
        ]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    prep_resize_node = ComposableNode(
        name='PreproccessingResizeNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/image'),
                    ('camera_info', 'data_loader/camera_info'),
                    ('resize/image', 'buffer/image'),
                    ('resize/camera_info', 'buffer/camera_info')],
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8', 'nitros_camera_info'],
        }],
        remappings=[('buffer/input0', 'buffer/image'),
                    ('input0', 'image_raw'),
                    ('buffer/input1', 'buffer/camera_info'),
                    ('input1', 'camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'isaac_ros_apriltag_interfaces/msg/AprilTagDetectionArray',
            'use_nitros_type_monitor_sub': False,
        }],
        remappings=[
            ('output', 'apriltag_detections')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSAprilTagGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_resize_node,
            playback_node,
            monitor_node,
            rectify_node,
            apriltag_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    return TestIsaacROSAprilTagGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSAprilTagGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac AprilTag graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac AprilTag Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # The slice of the rosbag to use
        input_data_start_time=3.0,
        input_data_end_time=3.5,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=600.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    def test_benchmark(self):
        self.run_benchmark()
