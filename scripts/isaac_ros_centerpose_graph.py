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
    3. CenterPoseDecoderNode:  turns tensors into 3d detections

Required:
- Packages:
    - isaac_ros_dnn_image_encoder
    - isaac_ros_tensor_rt
    - isaac_ros_centerpose
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
- Models:
    - assets/models/centerpose_shoe/centerpose_shoe.onnx
    - assets/models/centerpose_shoe/config.pbtxt
"""

import os
import shutil
import time

from ament_index_python.packages import get_package_share_directory
from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest
from ros2_benchmark import Resolution

IMAGE_RESOLUTION = Resolution(1920, 1200)
NETWORK_RESOLUTION = Resolution(512, 512)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_storage'
MODEL_NAME = 'centerpose_shoe'
MODEL_CONFIG_FILE_NAME = 'centerpose_shoe/config.pbtxt'
ENGINE_ROOT = '/tmp/models'
ENGINE_FILE_DIR = '/tmp/models/centerpose_shoe'
ENGINE_FILE_PATH = '/tmp/models/centerpose_shoe/1/model.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS Pose Estimation (CenterPose)."""
    MODELS_ROOT = os.path.join(TestIsaacROSCenterPose.get_assets_root_path(), 'models')

    centerpose_encoder_node = ComposableNode(
        name='DnnImageEncoderNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'input_image_width': IMAGE_RESOLUTION['width'],
            'input_image_height': IMAGE_RESOLUTION['height'],
            'network_image_width': NETWORK_RESOLUTION['width'],
            'network_image_height': NETWORK_RESOLUTION['height'],
            'image_mean': [0.408, 0.447, 0.47],
            'image_stddev': [0.289, 0.274, 0.278]
        }],
        remappings=[('encoded_tensor', 'tensor_pub')])

    centerpose_inference_node = ComposableNode(
        name='TritonNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': [ENGINE_ROOT],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['bboxes', 'scores', 'kps', 'clses',
                                    'obj_scale', 'kps_displacement_mean',
                                    'kps_heatmap_mean'],
            'output_binding_names': ['bboxes', 'scores', 'kps', 'clses',
                                     'obj_scale', 'kps_displacement_mean',
                                     'kps_heatmap_mean'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
        }])

    centerpose_decoder_node = ComposableNode(
            name='CenterPoseDecoderNode',
            namespace=TestIsaacROSCenterPose.generate_namespace(),
            package='isaac_ros_centerpose',
            plugin='nvidia::isaac_ros::centerpose::CenterPoseDecoderNode',
            parameters=[{
                'camera_matrix': [
                    651.2994384765625, 0.0, 298.3225504557292,
                    0.0, 651.2994384765625, 392.1635182698568,
                    0.0, 0.0, 1.0],
                'original_image_size': [600, 800],
                'output_field_size': [128, 128],
                'cuboid_scaling_factor': 1.0,
                'score_threshold': 0.3,
                'object_name': 'shoe',
            }],
            remappings=[('centerpose/detections', 'poses')],
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/camera_info')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSCenterPose.generate_namespace(),
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
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_detection3_d_array',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'poses')],
    )

    composable_node_container = ComposableNodeContainer(
        name='centerpose_container',
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            centerpose_encoder_node,
            centerpose_inference_node,
            centerpose_decoder_node,
        ],
        namespace=TestIsaacROSCenterPose.generate_namespace(),
        output='screen',
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSCenterPose.get_assets_root_path(), 'models')
    if not os.path.exists(os.path.dirname(ENGINE_FILE_PATH)):
        os.makedirs(os.path.dirname(ENGINE_FILE_PATH))
    shutil.copy(
        os.path.join(MODELS_ROOT, MODEL_CONFIG_FILE_NAME),
        ENGINE_FILE_DIR)

    # Generate engine file using trt-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
       trt_converter_args = [
           f'--onnx={MODELS_ROOT}/{MODEL_NAME}/centerpose_shoe.onnx',
           f'--saveEngine={ENGINE_FILE_PATH}',
           '--fp16'
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
        # Note that if the model failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for TensorRT Node to be launched
        time.sleep(self.TENSOR_RT_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
