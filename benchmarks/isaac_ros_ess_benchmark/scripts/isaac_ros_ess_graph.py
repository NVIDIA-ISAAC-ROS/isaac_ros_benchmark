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
Performance test for the Isaac ROS ESS stereo image graph.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. ImageFormatConverter, Resize, ImageNormalize, ImageToTensor, InterleavedToPlanar,
       Reshape nodes: Turns raw images into appropriately-shaped tensors
    2. TensorPairSyncNode: Syncs left and right tensors for TensorRT inference
    3. TensorRTNode: Runs TensorRT inference
    4. DNNStereoDecoderNode: Decodes disparity from TensorRT output
    5. PointCloudNode: Converts disparity to pointcloud

Required:
- Packages:
    - isaac_ros_ess
    - isaac_ros_dnn_stereo_decoder
    - isaac_ros_stereo_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hideaway
- Models:
    - assets/models/ess/ess.onnx
"""

import os
import time

import isaac_ros_ess_benchmark.ess_model_utility as ess_model_utility

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hideaway'
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 576


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS ESS graph."""
    asset_models_path = os.path.join(TestIsaacROSEssStereoGraph.get_assets_root_path(), 'models')
    _, engine_file_path = ess_model_utility.get_model_paths('full', asset_models_path)
    ess_plugin_path = ess_model_utility.get_plugin_path(asset_models_path)

    namespace = TestIsaacROSEssStereoGraph.generate_namespace()

    pipeline_nodes = ess_model_utility.create_ess_pipeline_nodes(
        namespace, NETWORK_WIDTH, NETWORK_HEIGHT, engine_file_path, ess_plugin_path)

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=namespace,
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
                    ('hawk_0_right_rgb_image', 'data_loader/right_image'),
                    ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=namespace,
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferPlaybackNode',
        parameters=[{
            'data_formats': [
                'sensor_msgs/msg/Image',
                'sensor_msgs/msg/CameraInfo',
                'sensor_msgs/msg/Image',
                'sensor_msgs/msg/CameraInfo'
            ],
        }],
        remappings=[('buffer/input0', 'data_loader/left_image'),
                    ('input0', 'left/image_rect'),
                    ('buffer/input1', 'data_loader/left_camera_info'),
                    ('input1', 'left/camera_info_rect'),
                    ('buffer/input2', 'data_loader/right_image'),
                    ('input2', 'right/image_rect'),
                    ('buffer/input3', 'data_loader/right_camera_info'),
                    ('input3', 'right/camera_info_rect')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=namespace,
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferMonitorNode',
        parameters=[{
            'monitor_data_format': 'sensor_msgs/msg/PointCloud2',
        }],
        remappings=[('output', 'points2')],
    )

    pointcloud_node = ComposableNode(
        name='PointCloudNode',
        namespace=namespace,
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'approximate_sync': False,
            'use_color': False,
            'use_system_default_qos': True,
        }],
        remappings=[
            ('left/image_rect_color', 'left/image_resize'),
            ('left/camera_info', 'left/camera_info_resize'),
            ('right/camera_info', 'right/camera_info_resize'),
        ]
    )

    composable_node_container = ComposableNodeContainer(
        name='ess_disparity_container',
        namespace=namespace,
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            *pipeline_nodes,
            pointcloud_node,
        ],
        output='screen',
    )

    return [composable_node_container]


def generate_test_description():
    asset_models_path = os.path.join(TestIsaacROSEssStereoGraph.get_assets_root_path(), 'models')
    ess_model_utility.generate_ess_engine_file('full', asset_models_path)
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
        playback_message_buffer_size=10,
        pre_trial_run_wait_time_sec=5.0,
    )

    # Amount of seconds to wait for ESS engine to be initialized
    ESS_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exists only if previous model conversion succeeds.
        # Note that if the model fails to be converted, an exception will be raised and
        # the entire test will end.
        asset_models_path = os.path.join(
            TestIsaacROSEssStereoGraph.get_assets_root_path(), 'models')
        while not ess_model_utility.is_ess_engine_file_generated('full', asset_models_path):
            time.sleep(1)
        # Wait for ESS Node to be launched
        time.sleep(self.ESS_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
