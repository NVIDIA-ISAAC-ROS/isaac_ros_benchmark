# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for Isaac ROS RT-DETR graph.

This test uses the SyntheticaDETR plan file trained on YCB + Whole Foods objects.
The graph consists of the following:
- Graph under Test:
    1. Resize, Pad, ImageFormatConverter, ImageToTensor, InterleavedToPlanar, Reshape Nodes:
         turns raw images into appropriately-shaped tensors
    2. RtDetrPreprocessorNode: attaches required image size tensor to encoded image tensor
    3. TritonNode: runs SyntheticaDETR to detect grocery objects
    3. RtDetrDecoderNode:  turns tensors into detection arrays

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_tensor_rt
    - isaac_ros_rtdetr

- Datasets:
    - assets/datasets/r2b_dataset/r2b_robotarm
- Models:
    - assets/models/sdetr/sdetr_grasp.onnx
"""

import os

from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.HD
NETWORK_SIZE = 640  # RT-DETR architecture requires square network resolution
NETWORK_RESOLUTION = Resolution(NETWORK_SIZE, NETWORK_SIZE)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_robotarm'
MODEL_FILE_NAME = 'sdetr/sdetr_grasp.onnx'
ENGINE_FILE_PATH = '/tmp/sdetr_grasp.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS RT-DETR graph."""
    resize_node = ComposableNode(
        name='ResizeNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height'],
            'output_width': NETWORK_RESOLUTION['width'],
            'output_height': NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }]
    )

    pad_node = ComposableNode(
        name='PadNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': NETWORK_RESOLUTION['width'],
            'output_image_height': NETWORK_RESOLUTION['height'],
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[(
            'image', 'resize/image'
        )]
    )

    image_format_converter_node = ComposableNode(
        name='ImageFormatConverter',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': NETWORK_RESOLUTION['width'],
                'image_height': NETWORK_RESOLUTION['height']
        }],
        remappings=[
            ('image_raw', 'padded_image'),
            ('image', 'image_rgb')]
    )

    image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'image_rgb'),
            ('tensor', 'normalized_tensor'),
        ]
    )

    interleave_to_planar_node = ComposableNode(
        name='InterleavedToPlanarNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [NETWORK_RESOLUTION['width'], NETWORK_RESOLUTION['height'], 3]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )

    reshape_node = ComposableNode(
        name='ReshapeNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [
                1, 3, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='RtdetrPreprocessor',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{
            'image_size': NETWORK_RESOLUTION['width']
        }],
        remappings=[
            ('encoded_tensor', 'reshaped_tensor')
        ]
    )

    tensor_rt_node = ComposableNode(
        name='TensorRt',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    rtdetr_decoder_node = ComposableNode(
        name='RtdetrDecoder',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info')
        ]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8', 'nitros_camera_info'],
        }],
        remappings=[
            ('buffer/input0', 'data_loader/image_raw'),
            ('input0', 'image'),
            ('buffer/input1', 'data_loader/camera_info'),
            ('input1', 'camera_info')
        ]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::MonitorNode',
        parameters=[{
            'monitor_data_format': 'vision_msgs/msg/Detection2DArray',
        }],
        remappings=[
            ('output', 'detections_output')
        ],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSRtDetr.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node, playback_node,
            resize_node, pad_node, image_format_converter_node,
            image_to_tensor_node, interleave_to_planar_node, reshape_node,
            rtdetr_preprocessor_node, tensor_rt_node, rtdetr_decoder_node,
            monitor_node
        ],
        output='screen',
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSRtDetr.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using trtexec
    if not os.path.isfile(ENGINE_FILE_PATH):
        trtexec_args = [
            f'--onnx={MODEL_FILE_PATH}',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16',
            '--skipInference',
        ]
        TRTConverter()(trtexec_args)
    return TestIsaacROSRtDetr.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSRtDetr(ROS2BenchmarkTest):
    """Performance test for Isaac ROS RT-DETR graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS RT-DETR Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': NETWORK_RESOLUTION
        }
    )

    def test_benchmark(self):
        self.run_benchmark()
