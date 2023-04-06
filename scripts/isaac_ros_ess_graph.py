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
Performance test for the Isaac ROS ESS stereo image graph.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. ESSDisparityNode: creates disparity images from stereo pair
    2. PointCloudNode: converts disparity to pointcloud

Required:
- Packages:
    - isaac_ros_ess
    - isaac_ros_stereo_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hideaway
- Models:
    - assets/models/ess/ess.etlt
"""

import os
import time

from isaac_ros_benchmark import TaoConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hideaway'
MODEL_FILE_NAME = 'ess/ess.etlt'
ENGINE_FILE_PATH = '/tmp/ess.engine'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS ESS graph."""

    disparity_node = ComposableNode(
        name='ESSDisparityNode',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_disparity::ESSDisparityNode',
        parameters=[{'engine_file_path': ENGINE_FILE_PATH}]
    )

    pointcloud_node = ComposableNode(
        name='PointCloudNode',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'approximate_sync': False,
                'use_color': False,
                'use_system_default_qos': True,
        }],
        remappings=[('left/image_rect_color', 'left/image_rect')])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
                    ('hawk_0_right_rgb_image', 'data_loader/right_image'),
                    ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
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
        remappings=[('buffer/input0', 'data_loader/left_image'),
                    ('input0', 'left/image_rect'),
                    ('buffer/input1', 'data_loader/left_camera_info'),
                    ('input1', 'left/camera_info'),
                    ('buffer/input2', 'data_loader/right_image'),
                    ('input2', 'right/image_rect'),
                    ('buffer/input3', 'data_loader/right_camera_info'),
                    ('input3', 'right/camera_info')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'points2')],
    )

    composable_node_container = ComposableNodeContainer(
        name='ess_disparity_container',
        namespace=TestIsaacROSEssStereoGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            disparity_node,
            pointcloud_node
        ],
        output='screen',
    )

    return [composable_node_container]

def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSEssStereoGraph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using tao-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        tao_converter_args = [
            '-k', 'ess',
            '-t', 'fp16',
            '-e', ENGINE_FILE_PATH,
            '-o', 'output_left', MODEL_FILE_PATH
        ]
        TaoConverter()(tao_converter_args)
    return TestIsaacROSEssStereoGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSEssStereoGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS ESS stereo image graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS ESS Stereo Image Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=350.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10
    )

    # Amount of seconds to wait for Triton Engine to be initialized
    ESS_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for ESS Node to be launched
        time.sleep(self.ESS_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
