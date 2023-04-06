# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for Isaac ROS CenterPose Pose Estimation.

This test uses the ONNX CenterPose model trained on images of shoes.
The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns raw images into resized, normalized tensors
    2. TensorRTNode: runs CenterPose model to estimate the 6DOF pose of target objects
    3. CenterPoseDecoderNode:  turns tensors into marker arrays

Required:
- Packages:
    - isaac_ros_dnn_encoders
    - isaac_ros_tensor_rt
    - isaac_ros_centerpose
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
- Models:
    - assets/models/shoe_resnet_140.onnx
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = Resolution(512, 512)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_storage'
MODEL_FILE_NAME = 'shoe_resnet_140.onnx'
ENGINE_FILE_PATH = '/tmp/shoe_resnet_140_engine.plan'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS Pose Estimation (CenterPose)."""

    config = os.path.join(
        get_package_share_directory('isaac_ros_benchmark'),
        'config',
        'centerpose_decoder_params.yaml'
    )

    centerpose_encoder_node = ComposableNode(
        name='DnnImageEncoderNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_dnn_encoders',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': IMAGE_RESOLUTION['width'],
            'network_image_height': IMAGE_RESOLUTION['height'],
            'image_mean': [0.408, 0.447, 0.47],
            'image_stddev': [0.289, 0.274, 0.278]
        }],
        remappings=[('encoded_tensor', 'tensor_pub')])

    centerpose_inference_node = ComposableNode(
        name='TensorRTNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': ENGINE_FILE_PATH,
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['hm', 'wh', 'hps', 'reg', 'hm_hp', 'hp_offset', 'scale'],
            'output_binding_names': ['hm', 'wh', 'hps', 'reg', 'hm_hp', 'hp_offset', 'scale'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
            'verbose': False,
            'force_engine_update': False,
        }])

    centerpose_decoder_node = Node(
            name='CenterPoseDecoderNode',
            namespace=TestIsaacROSCenterPose.generate_namespace(),
            package='isaac_ros_centerpose',
            executable='CenterPoseDecoder',
            parameters=[config],
            remappings=[('object_poses', 'poses')],
            output='screen'
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['sensor_msgs/msg/Image'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'visualization_msgs/msg/MarkerArray',
        }],
        remappings=[
            ('output', 'poses')],
    )

    composable_node_container = ComposableNodeContainer(
        name='dope_container',
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            centerpose_encoder_node,
            centerpose_inference_node
        ],
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        output='screen',
    )

    return [composable_node_container, centerpose_decoder_node]

def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSCenterPose.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using trt-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        trt_converter_args = [
            f'--onnx={MODEL_FILE_PATH}',
            f'--saveEngine={ENGINE_FILE_PATH}'
        ]
        TRTConverter()(trt_converter_args)

    return TestIsaacROSCenterPose.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSCenterPose(ROS2BenchmarkTest):
    """Performance test for Isaac CenterPose Pose Estimation."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS CenterPose Benchmark',
        input_data_path=ROSBAG_PATH,
        # The slice of the rosbag to use
        input_data_start_time=9.8,
        input_data_end_time=9.9,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=50.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    # Amount of seconds to wait for TensorRT Engine to be initialized
    TENSOR_RT_WAIT_SEC = 60

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for TensorRT Node to be launched
        time.sleep(self.TENSOR_RT_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()