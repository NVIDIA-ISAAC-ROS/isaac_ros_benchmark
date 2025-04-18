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
Performance test for the Isaac ROS Triton node with PeopleSemSegnet.

The graph consists of the following:
- Preprocessors:
    1. PrepDnnImageEncoderNode: turns raw images into resized, normalized tensors
- Graph under Test:
    1. TritonNode: runs PeopleSemSegnet to detect "person" objects

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
    - isaac_ros_triton
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
- Models:
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg.onnx
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg_cache.txt
    - assets/models/peoplesemsegnet_shuffleseg/config.pbtxt
"""

import os
import shutil
import time

from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

PLAYBACK_MESSAGE_BUFFER_SIZE = 1
IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = Resolution(960, 544)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
MODEL_NAME = 'peoplesemsegnet_shuffleseg'
MODEL_CONFIG_FILE_NAME = 'peoplesemsegnet_shuffleseg/config.pbtxt'
ENGINE_ROOT = '/tmp/models'
ENGINE_FILE_DIR = '/tmp/models/peoplesemsegnet_shuffleseg'
ENGINE_FILE_PATH = '/tmp/models/peoplesemsegnet_shuffleseg/1/model.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description with the Triton ROS 2 node for testing."""
    triton_node = ComposableNode(
        name='TritonNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': [ENGINE_ROOT],
            'max_batch_size': 0,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_2'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output'],
            'output_binding_names': ['argmax_1'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
        }],
        remappings=[('tensor_pub', 'input'), ('tensor_sub', 'output')]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    resize_node = ComposableNode(
        name='ResizeNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height'],
            'output_width': NETWORK_RESOLUTION['width'],
            'output_height': NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8'
        }],
        remappings=[
            ('image', 'data_loader/image'),
            ('camera_info', 'data_loader/camera_info')
        ]
    )

    image_format_converter_node = ComposableNode(
        name='ImageFormatConverter',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': NETWORK_RESOLUTION['width'],
                'image_height': NETWORK_RESOLUTION['height']
        }],
        remappings=[
            ('image_raw', 'resize/image'),
            ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'image_rgb')
        ]
    )

    normalize_node = ComposableNode(
        name='NormalizeNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': [0.5, 0.5, 0.5],
            'stddev': [0.5, 0.5, 0.5],
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }]
    )

    reshape_node = ComposableNode(
        name='ReshapeNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width'], 3],
            'output_tensor_shape': [
                1, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width'], 3]
        }],
        remappings=[
            ('tensor', 'normalized_tensor'),
            ('reshaped_tensor', 'buffer/input')
        ],
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
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
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_tensor_list_nhwc_rgb_f32',
            'use_nitros_type_monitor_sub': True,
        }],
    )

    composable_node_container = ComposableNodeContainer(
        name='triton_container',
        namespace=TestIsaacROSTritonNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            resize_node, image_format_converter_node,
            image_to_tensor_node, normalize_node, reshape_node,
            playback_node,
            triton_node,
            monitor_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSTritonNode.get_assets_root_path(), 'models')
    if not os.path.exists(os.path.dirname(ENGINE_FILE_PATH)):
        os.makedirs(os.path.dirname(ENGINE_FILE_PATH))
    shutil.copy(
        os.path.join(MODELS_ROOT, MODEL_CONFIG_FILE_NAME),
        ENGINE_FILE_DIR)

    # Generate engine file using trtexec
    if not os.path.isfile(ENGINE_FILE_PATH):
        trtexec_args = [
            f'--onnx={MODELS_ROOT}/{MODEL_NAME}/peoplesemsegnet_shuffleseg.onnx',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16',
            '--minShapes=input_2:1x544x960x3',
            '--optShapes=input_2:1x544x960x3',
            '--maxShapes=input_2:16x544x960x3',
            '--skipInference',
        ]
        TRTConverter()(trtexec_args)
    return TestIsaacROSTritonNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSTritonNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Triton node with PeopleSemSegnet."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS TritonNode (PeopleSemSegnet) Benchmark',
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

    # Amount of seconds to wait for Triton Engine to be initialized
    TRITON_WAIT_SEC = 90

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for Triton Node to be launched
        time.sleep(self.TRITON_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
