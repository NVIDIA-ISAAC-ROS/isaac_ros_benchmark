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
Performance test for the Isaac ROS TensorRT node with PeopleSemSegnet.

The graph consists of the following:
- Preprocessors:
    1. PrepDnnImageEncoderNode: turns raw images into resized, normalized tensors
- Graph under Test:
    1. TensorRTNode: runs PeopleSemSegnet to detect "person" objects

Required:
- Packages:
    - isaac_ros_tensor_rt
    - isaac_ros_dnn_image_encoder
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
- Models:
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg_etlt.etlt
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg_cache.txt
"""

import os
import time

from isaac_ros_benchmark import TaoConverter

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest
from ros2_benchmark import Resolution

PLAYBACK_MESSAGE_BUFFER_SIZE = 1
IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = Resolution(960, 544)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
MODEL_NAME = 'peoplesemsegnet_shuffleseg'
ENGINE_FILE_PATH = '/tmp/peoplesemsegnet_shuffleseg.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description with the TensorRT ROS 2 node for testing."""
    encoder_num_blocks = PLAYBACK_MESSAGE_BUFFER_SIZE * 2

    tensor_rt_node = ComposableNode(
        name='TensorRTNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        remappings=[('tensor_pub', 'input'),
                    ('tensor_sub', 'output')],
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'output_binding_names': ['argmax_1'],
            'output_tensor_names': ['output'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_2:0'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'verbose': False,
            'force_engine_update': False,
        }]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image')]
    )

    prep_encoder_node = ComposableNode(
        name='PrepDnnImageEncoderNode',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'input_image_width': IMAGE_RESOLUTION['width'],
            'input_image_height': IMAGE_RESOLUTION['height'],
            'network_image_width': NETWORK_RESOLUTION['width'],
            'network_image_height': NETWORK_RESOLUTION['height'],
            'network_image_encoding': 'rgb8',
            'image_mean': [0.5, 0.5, 0.5],
            'image_stddev': [0.5, 0.5, 0.5],
            'num_blocks': encoder_num_blocks
        }],
        remappings=[('image', 'data_loader/image'),
                    ('encoded_tensor', 'buffer/input')]
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
        name='tensor_rt_container',
        namespace=TestIsaacROSTensorRTNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_encoder_node,
            playback_node,
            monitor_node,
            tensor_rt_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSTensorRTNode.get_assets_root_path(), 'models')
    MODEL_DIR = os.path.join(MODELS_ROOT, MODEL_NAME)

    # Generate engine file using trt-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        tao_converter_args = [
            '-k', 'tlt_encode',
            '-p', 'input_2:0,1x3x544x960,1x3x544x960,1x3x544x960',
            '-t', 'int8',
            '-c', f'{MODEL_DIR}/peoplesemsegnet_shuffleseg_cache.txt',
            '-e', ENGINE_FILE_PATH,
            '-o', 'argmax_1',
            f'{MODEL_DIR}/peoplesemsegnet_shuffleseg_etlt.etlt'
        ]
        TaoConverter()(tao_converter_args)
    return TestIsaacROSTensorRTNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSTensorRTNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS TensorRT node with PeopleSemSegnet."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS TensorRTNode (PeopleSemSegnet) Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=3000.0,
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
