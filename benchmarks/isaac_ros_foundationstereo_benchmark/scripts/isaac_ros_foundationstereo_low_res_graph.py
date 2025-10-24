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
Performance test for the Isaac ROS FoundationStereo graph using the low-res (320x736) model.

The graph consists of the following:
- Graph under Test:
    1. Resize, Pad, ImageFormatConverter, ImageNormalize, ImageToTensor, InterleavedToPlanar,
       Reshape Nodes: Turns raw images into appropriately-shaped tensors
    2. TensorPairSyncNode: Syncs left and right tensors and prepares them for TensorRT inference
    3. TensorRTNode: Runs TensorRT inference
    4. FoundationStereoDecoderNode: Decodes disparity from TensorRT output
    5. PointCloudNode: Converts disparity to pointcloud

Required:
- Packages:
    - isaac_ros_foundationstereo
    - isaac_ros_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_tensor_rt
    - isaac_ros_stereo_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hideaway
- Models:
    - assets/models/foundationstereo/foundationstereo_320x736.onnx
"""

import os

from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

INPUT_WIDTH = 736
INPUT_HEIGHT = 320
IMAGE_RESOLUTION = Resolution(INPUT_WIDTH, INPUT_HEIGHT)
NETWORK_WIDTH = 736
NETWORK_HEIGHT = 320
NETWORK_RESOLUTION = Resolution(NETWORK_WIDTH, NETWORK_HEIGHT)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hideaway'
MODEL_FILE_NAME = 'foundationstereo/foundationstereo_320x736.onnx'
ENGINE_FILE_PATH = '/tmp/foundationstereo_320x736.engine'
MODEL_NUM_CHANNELS = 3


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS FoundationStereo graph."""
    # Data loader and playback nodes
    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
                    ('hawk_0_right_rgb_image', 'data_loader/right_image'),
                    ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
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
                    ('input1', 'left/camera_info_rect'),
                    ('buffer/input2', 'data_loader/right_image'),
                    ('input2', 'right/image_rect'),
                    ('buffer/input3', 'data_loader/right_camera_info'),
                    ('input3', 'right/camera_info_rect')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'points2')],
    )

    # --- FoundationStereo pipeline nodes (left and right) ---
    # Left image pipeline
    left_resize_node = ComposableNode(
        name='left_resize_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
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
        remappings=[
            ('image', 'left/image_rect'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )
    left_pad_node = ComposableNode(
        name='left_pad_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': NETWORK_RESOLUTION['width'],
            'output_image_height': NETWORK_RESOLUTION['height'],
            'border_type': 'REPLICATE'
        }],
        remappings=[
            ('image', 'left/image_resize'),
            ('padded_image', 'left/image_pad'),
        ]
    )
    left_format_node = ComposableNode(
        name='left_format_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': NETWORK_RESOLUTION['width'],
            'image_height': NETWORK_RESOLUTION['height'],
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'left/image_pad'),
            ('image', 'left/image_rgb')
        ]
    )
    left_normalize_node = ComposableNode(
        name='left_normalize_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('normalized_image', 'left/image_normalize')
        ]
    )
    left_tensor_node = ComposableNode(
        name='left_tensor_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'left_image',
        }],
        remappings=[
            ('image', 'left/image_normalize'),
            ('tensor', 'left/tensor'),
        ]
    )
    left_planar_node = ComposableNode(
        name='left_planar_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width'],
                                   MODEL_NUM_CHANNELS],
            'output_tensor_name': 'left_image'
        }],
        remappings=[
            ('interleaved_tensor', 'left/tensor'),
            ('planar_tensor', 'left/tensor_planar')
        ]
    )
    left_reshape_node = ComposableNode(
        name='left_reshape_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, NETWORK_RESOLUTION['height'],
                                   NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, NETWORK_RESOLUTION['height'],
                                    NETWORK_RESOLUTION['width']]
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    # Right image pipeline
    right_resize_node = ComposableNode(
        name='right_resize_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
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
        remappings=[
            ('image', 'right/image_rect'),
            ('camera_info', 'right/camera_info_rect'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )
    right_pad_node = ComposableNode(
        name='right_pad_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': NETWORK_RESOLUTION['width'],
            'output_image_height': NETWORK_RESOLUTION['height'],
            'border_type': 'REPLICATE'
        }],
        remappings=[
            ('image', 'right/image_resize'),
            ('padded_image', 'right/image_pad'),
        ]
    )
    right_format_node = ComposableNode(
        name='right_format_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': NETWORK_RESOLUTION['width'],
            'image_height': NETWORK_RESOLUTION['height'],
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_pad'),
            ('image', 'right/image_rgb')
        ]
    )
    right_normalize_node = ComposableNode(
        name='right_normalize_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [123.675, 116.28, 103.53],
            'stddev': [58.395, 57.12, 57.375],
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('normalized_image', 'right/image_normalize')
        ]
    )
    right_tensor_node = ComposableNode(
        name='right_tensor_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'right_image',
        }],
        remappings=[
            ('image', 'right/image_normalize'),
            ('tensor', 'right/tensor'),
        ]
    )
    right_planar_node = ComposableNode(
        name='right_planar_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [NETWORK_RESOLUTION['height'], NETWORK_RESOLUTION['width'],
                                   MODEL_NUM_CHANNELS],
            'output_tensor_name': 'right_image'
        }],
        remappings=[
            ('interleaved_tensor', 'right/tensor'),
            ('planar_tensor', 'right/tensor_planar')
        ]
    )
    right_reshape_node = ComposableNode(
        name='right_reshape_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, NETWORK_RESOLUTION['height'],
                                   NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, NETWORK_RESOLUTION['height'],
                                    NETWORK_RESOLUTION['width']]
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    )

    # Tensor sync node
    tensor_pair_sync_node = ComposableNode(
        name='tensor_pair_sync_node',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image',
            'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'left_image',
            'output_tensor2_name': 'right_image'
        }],
        remappings=[
            ('tensor1', 'left/tensor_reshape'),
            ('tensor2', 'right/tensor_reshape'),
        ]
    )

    # TensorRT node
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': ['left_image', 'right_image'],
            'input_binding_names': ['left_image', 'right_image'],
            'output_tensor_names': ['disparity'],
            'output_binding_names': ['disparity'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    # Disparity decoder node
    foundationstereo_decoder_node = ComposableNode(
        name='foundationstereo_decoder',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_foundationstereo',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::FoundationStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'disparity'
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_resize')
        ]
    )

    # PointCloud node (optional, if you want to output pointclouds)
    pointcloud_node = ComposableNode(
        name='PointCloudNode',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
            'approximate_sync': False,
            'use_color': False,
            'use_system_default_qos': True,
        }],
        remappings=[('left/image_rect_color', 'left/image_pad'),
                    ('left/camera_info', 'left/camera_info_resize'),
                    ('right/camera_info', 'right/camera_info_resize')])

    composable_node_container = ComposableNodeContainer(
        name='foundationstereo_container',
        namespace=TestIsaacROSFoundationStereoGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            left_resize_node,
            left_pad_node,
            left_format_node,
            left_normalize_node,
            left_tensor_node,
            left_planar_node,
            left_reshape_node,
            right_resize_node,
            right_pad_node,
            right_format_node,
            right_normalize_node,
            right_tensor_node,
            right_planar_node,
            right_reshape_node,
            tensor_pair_sync_node,
            tensor_rt_node,
            foundationstereo_decoder_node,
            pointcloud_node
        ],
        output='screen',
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSFoundationStereoGraph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using trtexec
    if not os.path.isfile(ENGINE_FILE_PATH):
        trtexec_args = [
            f'--onnx={MODEL_FILE_PATH}',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16',
            '--skipInference'
        ]
        TRTConverter()(trtexec_args)

    return TestIsaacROSFoundationStereoGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSFoundationStereoGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS FoundationStereo graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS FoundationStereo Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=20.0,
        publisher_lower_frequency=1.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        pre_trial_run_wait_time_sec=5.0,
        custom_report_info={
            'network_resolution': NETWORK_RESOLUTION
        }
    )

    def test_benchmark(self):
        self.run_benchmark()
