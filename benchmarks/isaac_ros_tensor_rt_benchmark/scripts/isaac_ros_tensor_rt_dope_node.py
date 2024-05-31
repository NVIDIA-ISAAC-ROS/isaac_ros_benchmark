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
Performance test for the Isaac ROS TensorRT node.

The graph consists of the following:
- Preprocessors:
    1. PrepDnnImageEncoderNode: turns raw images into resized, normalized tensors
- Graph under Test:
    1. TensorRTNode: converts an input tensor into a tensor of belief map

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
    - isaac_ros_tensor_rt
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hope
- Models:
    - assets/models/ketchup/ketchup.onnx
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_benchmark import TRTConverter
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

PLAYBACK_MESSAGE_BUFFER_SIZE = 1
IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = ImageResolution.VGA
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hope'
MODEL_FILE_NAME = 'ketchup/ketchup.onnx'
ENGINE_FILE_PATH = '/tmp/ketchup_engine.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description with the TensorRT ROS 2 node for testing."""
    tensor_rt_node = ComposableNode(
        name='TensorRTNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[('tensor_pub', 'input'),
                    ('tensor_sub', 'output')],
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_binding_names': ['output'],
            'output_tensor_names': ['output'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('image', 'data_loader/image'),
                    ('camera_info', 'data_loader/camera_info')]
    )

    encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    encoder_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
        ),
        launch_arguments={
            'input_image_width': str(IMAGE_RESOLUTION['width']),
            'input_image_height': str(IMAGE_RESOLUTION['height']),
            'network_image_width': str(NETWORK_RESOLUTION['width']),
            'network_image_height': str(NETWORK_RESOLUTION['height']),
            'image_mean': str([0.5, 0.5, 0.5]),
            'image_stddev': str([0.5, 0.5, 0.5]),
            'encoding_desired': 'rgb8',
            'image_input_topic': 'data_loader/image',
            'camera_info_input_topic': 'data_loader/camera_info',
            'tensor_output_topic': 'buffer/input',
            'attach_to_shared_component_container': 'True',
            'component_container_name':
                f'{TestIsaacROSTensorRTNode.generate_namespace()}/tensor_rt_container',
            'dnn_image_encoder_namespace': TestIsaacROSTensorRTNode.generate_namespace(),
        }.items(),
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_tensor_list_nchw_rgb_f32',
            ],
        }],
        remappings=[('buffer/input0', 'buffer/input'),
                    ('input0', 'input')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_tensor_list_nhwc_rgb_f32',
            'use_nitros_type_monitor_sub': True,
        }],
    )

    composable_node_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='tensor_rt_container',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            tensor_rt_node
        ],
        output='screen'
    )

    return [composable_node_container, encoder_node_launch]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSTensorRTNode.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using trt-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        trt_converter_args = [
            f'--onnx={MODEL_FILE_PATH}',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16'
        ]
        TRTConverter()(trt_converter_args)
    return TestIsaacROSTensorRTNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSTensorRTNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS TensorRT node."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS TensorRTNode (DOPE) Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=350.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=PLAYBACK_MESSAGE_BUFFER_SIZE,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': NETWORK_RESOLUTION
        }
    )

    # Amount of seconds to wait for TensorRT Engine to be initialized
    TENSOR_RT_WAIT_SEC = 60

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for TensorRT Node to be launched
        time.sleep(self.TENSOR_RT_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
