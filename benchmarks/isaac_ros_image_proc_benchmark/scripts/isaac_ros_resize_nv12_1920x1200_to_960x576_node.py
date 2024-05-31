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
Performance test for the Isaac ROS resize node.

The graph consists of the following:
- Preprocessors:
    1. PrepImageConverterNode: converts image format to nv12
- Graph under Test:
    1. ResizeNode: resizes images

Required:
- Packages:
    - isaac_ros_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

SOURCE_IMAGE_RESOLUTION = ImageResolution.WUXGA
IMAGE_RESOLUTION = Resolution(960, 576)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_storage'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for isaac resize node."""
    resize_node = ComposableNode(
        name='ResizeNode',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'buffer/image'),
                    ('hawk_0_left_rgb_camera_info', 'buffer/camera_info')]
    )

    prep_image_format_converter_node = ComposableNode(
        name='PrepImageConverterNode',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'nv12',
            'image_width': SOURCE_IMAGE_RESOLUTION['width'],
            'image_height': SOURCE_IMAGE_RESOLUTION['height'],
        }],
        remappings=[
            ('image_raw', 'buffer/image'),
            ('image', 'buffer/image_converted')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_nv12', 'nitros_camera_info'],
        }],
        remappings=[('buffer/input0', 'buffer/image_converted'),
                    ('input0', 'image'),
                    ('buffer/input1', 'buffer/camera_info'),
                    ('input1', 'camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_image_nv12',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'resize/image')],
    )

    composable_node_container = ComposableNodeContainer(
        name='resize_container',
        namespace=TestIsaacROSResizeNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_image_format_converter_node,
            playback_node,
            resize_node,
            monitor_node,
        ],
        output='screen',
    )

    return [composable_node_container]


def generate_test_description():
    return TestIsaacROSResizeNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSResizeNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS resize node."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS ResizeNode NV12 1920x1200 to 960x576 Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=2500.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10,
        custom_report_info={
            'data_resolution': SOURCE_IMAGE_RESOLUTION,
            'resized_resolution': IMAGE_RESOLUTION,
        }
    )

    def test_benchmark(self):
        self.run_benchmark()
