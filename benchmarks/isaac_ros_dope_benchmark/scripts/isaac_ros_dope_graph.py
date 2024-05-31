# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for Isaac ROS DOPE Pose Estimation.

The graph (without pose refinement) consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns a raw image into a resized, normalized tensor
    2. TensorRTNode: converts an input tensor into a tensor of belief map
    3. DopeDecoder: converts a belief map into an array of poses

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
    - isaac_ros_tensor_rt
    - isaac_ros_dope
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

IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = ImageResolution.VGA
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hope'
MODEL_FILE_NAME = 'ketchup/ketchup.onnx'
ENGINE_FILE_PATH = '/tmp/ketchup_engine.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Pose Estimation (DOPE)."""
    dnn_image_encoder_namespace = TestIsaacROSDeepObjectPoseEstimation.generate_namespace()
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
            'encoding_desired': 'rgb8',
            'tensor_output_topic': 'tensor_pub',
            'attach_to_shared_component_container': 'True',
            'component_container_name': f'{dnn_image_encoder_namespace}/container',
            'dnn_image_encoder_namespace': dnn_image_encoder_namespace,
        }.items(),
    )

    dope_inference_node = ComposableNode(
        name='TensorRTNode',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'output_tensor_names': ['output'],
            'output_binding_names': ['output'],
            'force_engine_update': False,
            'verbose': False
        }])

    dope_decoder_node = ComposableNode(
        name='DopeDecoder',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='isaac_ros_dope',
        plugin='nvidia::isaac_ros::dope::DopeDecoderNode',
        parameters=[{
            'frame_id': 'dope'
        }],
        remappings=[('belief_map_array', 'tensor_sub'),
                    ('dope/pose_array', 'poses')])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('image', 'data_loader/image'),
                    ('camera_info', 'data_loader/camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['sensor_msgs/msg/Image',
                             'sensor_msgs/msg/CameraInfo'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image'),
                    ('buffer/input1', 'data_loader/camera_info'),
                    ('input1', 'camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'geometry_msgs/msg/PoseArray',
        }],
        remappings=[
            ('output', 'poses')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSDeepObjectPoseEstimation.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            dope_inference_node,
            dope_decoder_node
        ],
        output='screen',
    )

    return [composable_node_container, encoder_node_launch]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSDeepObjectPoseEstimation.get_assets_root_path(),
                               'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using trt-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        trt_converter_args = [
            f'--onnx={MODEL_FILE_PATH}',
            f'--saveEngine={ENGINE_FILE_PATH}',
            '--fp16'
        ]
        TRTConverter()(trt_converter_args)
    return TestIsaacROSDeepObjectPoseEstimation.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSDeepObjectPoseEstimation(ROS2BenchmarkTest):
    """Performance test for Isaac ROS DOPE Pose Estimation."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Deep Object Pose Estimation (DOPE) Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=300.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': NETWORK_RESOLUTION
        }
    )

    # Amount of seconds to wait for TensorRT Engine to be initialized
    TENSOR_RT_WAIT_SEC = 60

    def test_benchmark(self):
        self.run_benchmark()

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for TensorRT Node to be launched
        time.sleep(self.TENSOR_RT_WAIT_SEC)
