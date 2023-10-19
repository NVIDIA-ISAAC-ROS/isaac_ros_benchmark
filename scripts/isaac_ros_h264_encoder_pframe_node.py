# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for the Isaac ROS encoder node (p-frame).

The graph consists of the following:
- Preprocessors:
    1. PrepResizeNode: resizes images to full HD
- Graph under Test:
    1. EncoderNode: compresses raw images (p-frame)

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_h264_encoder
- Datasets:
    - assets/datasets/r2b_dataset/r2b_mezzanine
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.FULL_HD
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_mezzanine'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS EncoderNode."""

    encoder_node = ComposableNode(
        name='EncoderNode',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        parameters=[{
                'input_width': IMAGE_RESOLUTION['width'],
                'input_height': IMAGE_RESOLUTION['height'],
                'config': 'pframe_cqp',
        }])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info')]
    )

    prep_resize_node = ComposableNode(
        name='PrepResizeNode',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/left_image'),
                    ('camera_info', 'data_loader/left_camera_info'),
                    ('resize/image', 'buffer/image_raw'),
                    ('resize/camera_info', 'buffer/camera_info')],
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8'],
        }],
        remappings=[('buffer/input0', 'buffer/image_raw'),
                    ('input0', 'image_raw')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_compressed_image',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'image_compressed')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSEncoderNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            prep_resize_node,
            monitor_node,
            encoder_node
        ],
        output='screen'
    )

    return [composable_node_container]

def generate_test_description():
    return TestIsaacROSEncoderNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSEncoderNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS encoder node (p-frame)."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS EncoderNode (P-Frame) Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=600.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    def test_benchmark(self):
        self.run_benchmark()
