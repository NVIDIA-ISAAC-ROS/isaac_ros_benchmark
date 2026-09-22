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
Performance test for the Isaac ROS DNN image encoder launch graph.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DNN image encoder launch graph: turns raw images into resized, normalized tensors

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
    """Generate launch description for benchmarking the DNN image encoder graph."""
    namespace = TestIsaacROSDnnImageEncoderGraph.generate_namespace()
    dnn_image_encoder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('isaac_ros_dnn_image_encoder'),
                'launch',
                'dnn_image_encoder.launch.py',
            )
        ),
        launch_arguments={
            'input_image_width': '1920',
            'input_image_height': '1080',
            'network_image_width': '512',
            'network_image_height': '512',
            'input_encoding': 'bgr8',
            'image_mean': '[0.5, 0.5, 0.5]',
            'image_stddev': '[0.5, 0.5, 0.5]',
            'enable_padding': 'True',
            'dnn_image_encoder_namespace': namespace,
            'image_input_topic': 'image',
            'camera_info_input_topic': 'camera_info',
            'tensor_output_topic': 'output',
            'attach_to_shared_component_container': 'True',
            'component_container_name': f'{namespace}/container',
        }.items(),
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=namespace,
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=namespace,
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferPlaybackNode',
        parameters=[{
            'data_formats': ['sensor_msgs/msg/Image', 'sensor_msgs/msg/CameraInfo'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image'),
                    ('buffer/input1', 'data_loader/camera_info'),
                    ('input1', 'camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=namespace,
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferMonitorNode',
        parameters=[{
            'monitor_data_format': 'isaac_ros_tensor_msgs/msg/TensorList',
        }],
        remappings=[
            ('output', 'output')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=namespace,
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

    return [composable_node_container, dnn_image_encoder_launch]


def generate_test_description():
    return TestIsaacROSDnnImageEncoderGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSDnnImageEncoderGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS DNN image encoder launch graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS DNN Image Encoder Graph Benchmark',
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
