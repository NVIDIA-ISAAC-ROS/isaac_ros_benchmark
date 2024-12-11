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
Performance test for the Isaac ROS U-Net graph.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns raw images into resized, normalized tensors
    2. TensorRTNode: runs PeopleSemSegnet to detect "person" objects
    3. UNetDecoderNode: converts inference results to segmentation masks

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
    - isaac_ros_tensor_rt
    - isaac_ros_unet
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
- Models:
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg.onnx
    - assets/models/peoplesemsegnet_shuffleseg/peoplesemsegnet_shuffleseg_cache.txt
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_benchmark import TRTConverter
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = Resolution(960, 544)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
MODEL_NAME = 'peoplesemsegnet_shuffleseg'
ENGINE_FILE_PATH = '/tmp/peoplesemsegnet_shuffleseg.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for U-Net."""
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
            'tensor_output_topic': 'tensor_pub',
            'attach_to_shared_component_container': 'True',
            'component_container_name':
                f'{TestIsaacROSUNetGraph.generate_namespace()}/unet_container',
            'dnn_image_encoder_namespace': TestIsaacROSUNetGraph.generate_namespace(),
        }.items(),
    )

    tensorrt_node = ComposableNode(
        name='TensorRTNode',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'output_binding_names': ['argmax_1'],
            'output_tensor_names': ['output'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_2:0'],
            'verbose': False,
            'force_engine_update': False,
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32']
        }])

    unet_decoder_node = ComposableNode(
        name='UNetDecoderNode',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='isaac_ros_unet',
        plugin='nvidia::isaac_ros::unet::UNetDecoderNode',
        parameters=[{
            'network_output_type': 'argmax',
            'color_segmentation_mask_encoding': 'rgb8',
            'color_palette': [0x556B2F, 0x800000, 0x008080, 0x000080, 0x9ACD32, 0xFF0000,
                              0xFF8C00, 0xFFD700, 0x00FF00, 0xBA55D3, 0x00FA9A, 0x00FFFF,
                              0x0000FF, 0xF08080, 0xFF00FF, 0x1E90FF, 0xDDA0DD, 0xFF1493,
                              0x87CEFA, 0xFFDEAD],
        }])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_rgb8', 'nitros_camera_info'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image'),
                    ('buffer/input1', 'data_loader/camera_info'),
                    ('input1', 'camera_info')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'unet/raw_segmentation_mask')],
    )

    composable_node_container = ComposableNodeContainer(
        name='unet_container',
        namespace=TestIsaacROSUNetGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            tensorrt_node,
            unet_decoder_node
        ],
        output='screen',
    )

    return [composable_node_container, encoder_node_launch]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSUNetGraph.get_assets_root_path(), 'models')
    MODEL_DIR = os.path.join(MODELS_ROOT, MODEL_NAME)

    # Generate engine file using trtexec
    if not os.path.isfile(ENGINE_FILE_PATH):
        trtexec_args = [
            '--maxShapes=input_2:0:1x3x544x960',
            '--minShapes=input_2:0:1x3x544x960',
            '--optShapes=input_2:0:1x3x544x960',
            f'--onnx={MODEL_DIR}/peoplesemsegnet_shuffleseg.onnx',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16',
            '--skipInference',
        ]
        TRTConverter()(trtexec_args)
    return TestIsaacROSUNetGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSUNetGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS U-Net graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS TensorRT U-Net Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=3000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': NETWORK_RESOLUTION
        }
    )

    # Amount of seconds to wait for U-Net to be initialized
    UNET_WAIT_SEC = 60

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for U-Net Node to be launched
        time.sleep(self.UNET_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
