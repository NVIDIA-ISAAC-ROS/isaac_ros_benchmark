# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for Isaac ROS Grounding DINO graph.

This test uses the Grounding DINO plan file trained on commercial data.
The graph consists of the following:
- Graph under Test:
    1. Resize, Pad, ImageToTensor, ImageTensorNormalize, InterleavedToPlanar,
         Reshape Nodes: turn raw images into appropriately-shaped tensors
    2. GroundingDinoPreprocessor, GroundingDinoTextTokenizer Nodes: tokenize input
         prompt and attache the encoded image tensor
    3. TensorRTNode: runs Grounding DINO to detect arbitrary objects
    4. GroundingDinoDecoderNode:  turns tensors into detection arrays

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_tensor_rt
    - isaac_ros_grounding_dino

- Datasets:
    - assets/datasets/r2b_dataset/r2b_robotarm
- Models:
    - assets/models/grounding_dino/grounding_dino_model.onnx
"""

import os

from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.HD
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 544
NETWORK_NUM_CHANNELS = 3
IMAGE_ENCODER_MEAN = [0.485, 0.456, 0.406]
IMAGE_ENCODER_STD = [0.229, 0.224, 0.225]
NETWORK_RESOLUTION = Resolution(NETWORK_WIDTH, NETWORK_HEIGHT)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_robotarm'
MODEL_FILE_NAME = 'grounding_dino/grounding_dino_model.onnx'
ENGINE_FILE_PATH = '/tmp/grounding_dino_model.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS Grounding DINO graph."""
    resize_node = ComposableNode(
        name='resize_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
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
        }],
    )

    pad_node = ComposableNode(
        name='pad_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
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

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': True,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'image_tensor'),
        ]
    )

    normalize_node = ComposableNode(
        name='normalize_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': IMAGE_ENCODER_MEAN,
            'stddev': IMAGE_ENCODER_STD,
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }],
        remappings=[
            ('tensor', 'image_tensor'),
        ],
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [
                NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width'], NETWORK_NUM_CHANNELS]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'images',
            'input_tensor_shape': [
                NETWORK_NUM_CHANNELS, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [
                1, NETWORK_NUM_CHANNELS, NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width']]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )

    text_tokenizer_node = Node(
        name='grounding_dino_text_tokenizer',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_grounding_dino',
        executable='isaac_ros_grounding_dino_text_tokenizer.py',
        output='screen',
    )

    grounding_dino_preprocessor = ComposableNode(
        name='grounding_dino_preprocessor',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoPreprocessorNode',
        parameters=[{
            'default_prompt': 'can.'
        }],
        remappings=[
            ('image_tensor', 'reshaped_tensor')
        ]
    )

    grounding_dino_inference_node = ComposableNode(
        name='grounding_dino_inference',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': [
                'images', 'input_ids', 'attention_mask', 'position_ids',
                'token_type_ids', 'text_token_mask'
            ],
            'input_binding_names': [
                'inputs', 'input_ids', 'attention_mask', 'position_ids',
                'token_type_ids', 'text_token_mask'
            ],
            'output_tensor_names': ['scores', 'boxes'],
            'output_binding_names': ['pred_logits', 'pred_boxes'],
            'verbose': False,
            'force_engine_update': False
        }],
    )

    grounding_dino_decoder_node = ComposableNode(
        name='grounding_dino_decoder',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='isaac_ros_grounding_dino',
        plugin='nvidia::isaac_ros::grounding_dino::GroundingDinoDecoderNode',
        parameters=[{
            'image_width': NETWORK_RESOLUTION['width'],
            'image_height': NETWORK_RESOLUTION['height'],
        }],
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info')
        ]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
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
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
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
        namespace=TestIsaacROSGroundingDino.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node, playback_node,
            resize_node, pad_node, image_to_tensor_node,
            normalize_node, interleave_to_planar_node, reshape_node,
            grounding_dino_preprocessor, grounding_dino_inference_node,
            grounding_dino_decoder_node,
            monitor_node
        ],
        output='screen',
    )

    return [composable_node_container, text_tokenizer_node]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSGroundingDino.get_assets_root_path(), 'models')
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
    return TestIsaacROSGroundingDino.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSGroundingDino(ROS2BenchmarkTest):
    """Performance test for Isaac ROS Grounding DINO graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Grounding DINO Graph Benchmark',
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
