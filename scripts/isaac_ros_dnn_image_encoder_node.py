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
Performance test for Isaac ROS DnnImageEncoderNode.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns raw images into resized, normalized tensors

Required:
- Packages:
    - isaac_ros_dnn_encoders
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
INPUT_TENSOR_DIMENSIONS = [1, 3, 1920, 1080]

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS DnnImageEncoderNode."""

    encoder_node = ComposableNode(
        name='DnnImageEncoderNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='isaac_ros_dnn_encoders',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            # If updated image dimensions and encoding, please also update performance
            # metrics values at end of benchmark
            'network_image_width': 640,
            'network_image_height': 480,
            'network_image_encoding': 'rgb8',
            'network_normalization_type': 'positive_negative'
        }],
        remappings=[('encoded_tensor', 'output')])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSDnnImageEncoderNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image')]
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
            encoder_node
        ],
        output='screen'
    )

    return [composable_node_container]

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
