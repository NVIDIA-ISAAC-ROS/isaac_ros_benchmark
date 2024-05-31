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
Performance test for Isaac ROS DnnImageEncoderNode.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns raw images into resized, normalized tensors

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ros2_benchmark import Resolution, ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
IMAGE_RESOLUTION = Resolution(1920, 1200)
INPUT_TENSOR_DIMENSIONS = [1, 3, IMAGE_RESOLUTION['width'], IMAGE_RESOLUTION['height']]


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS DnnImageEncoderNode."""
    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    encoder_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': str(IMAGE_RESOLUTION['width']),
            'input_image_height': str(IMAGE_RESOLUTION['height']),
            'network_image_width': str(640),
            'network_image_height': str(480),
            'encoding_desired': 'bgr8',
            'tensor_output_topic': 'output',
            'attach_to_shared_component_container': 'True',
            'component_container_name':
                f'{TestIsaacROSDnnImageEncoderNode.generate_namespace()}/container',
            'dnn_image_encoder_namespace': TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        }.items(),
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8', 'nitros_camera_info'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image'),
                    ('buffer/input1', 'data_loader/camera_info'),
                    ('input1', 'camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_tensor_list_nchw_rgb_f32',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'output')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
        ],
        output='screen'
    )

    return [composable_node_container, encoder_node_launch]


def generate_test_description():
    return TestIsaacROSDnnImageEncoderNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSDnnImageEncoderNode(ROS2BenchmarkTest):
    """Performance test for Isaac ROS DnnImageEncoderNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS DnnImageEncoderNode Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=6000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10,
        custom_report_info={'data_resolution': INPUT_TENSOR_DIMENSIONS}
    )

    def test_benchmark(self):
        self.run_benchmark()
