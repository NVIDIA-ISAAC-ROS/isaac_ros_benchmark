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
Performance test for Isaac ROS DetectNet graph.

This test uses the PeopleNet plan file trained on images of pedestrians.
The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. DnnImageEncoderNode: turns raw images into resized, normalized tensors
    2. TritonNode: runs PeopleNet to detect pedestrians
    3. DetectNetDecoderNode:  turns tensors into detection arrays

Required:
- Packages:
    - isaac_ros_dnn_encoders
    - isaac_ros_triton
    - isaac_ros_detectnet
- Datasets:
    - assets/datasets/r2b_dataset/r2b_hallway
- Models:
    - assets/models/peoplenet/resnet34_peoplenet_int8.etlt
    - assets/models/peoplenet/resnet34_peoplenet_int8.txt
    - assets/models/peoplenet/config.pbtxt
    - assets/models/peoplenet/labels.txt
"""

import os
import shutil
import time

from isaac_ros_benchmark import TaoConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = Resolution(960, 544)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_hallway'
MODEL_NAME = 'peoplenet'
MODEL_CONFIG_FILE_NAME = 'peoplenet/config.pbtxt'
ENGINE_ROOT = '/tmp/models'
ENGINE_FILE_DIR = '/tmp/models/peoplenet'
ENGINE_FILE_PATH = '/tmp/models/peoplenet/1/model.plan'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS DetectNet graph."""

    # Read labels from text file
    MODELS_ROOT = os.path.join(TesetIsaacROSDetectNet.get_assets_root_path(), 'models')
    LABELS_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_NAME, 'labels.txt')
    with open(LABELS_FILE_PATH, 'r') as fd:
        label_list = fd.read().strip().splitlines()

    encoder_node = ComposableNode(
        name='DnnImageEncoderNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='isaac_ros_dnn_encoders',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        parameters=[{
            'network_image_width': IMAGE_RESOLUTION['width'],
            'network_image_height': IMAGE_RESOLUTION['height']
        }],
        remappings=[('encoded_tensor', 'tensor_pub')]
    )

    triton_node = ComposableNode(
        name='TritonNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': [ENGINE_ROOT],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': ['input_1'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['output_cov', 'output_bbox'],
            'output_binding_names': ['output_cov/Sigmoid', 'output_bbox/BiasAdd'],
            'output_tensor_formats': ['nitros_tensor_list_nhwc_rgb_f32'],
        }]
    )

    detectnet_decoder_node = ComposableNode(
        name='DetectNetDecoderNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        parameters=[{
            'label_list': label_list,
            'enable_confidence_threshold': True,
            'enable_bbox_area_threshold': True,
            'enable_dbscan_clustering': True,
            'confidence_threshold': 0.35,
            'min_bbox_area': 100.0,
            'dbscan_confidence_threshold': 0.35,
            'dbscan_eps': 0.7,
            'dbscan_min_boxes': 1,
            'dbscan_enable_athr_filter': 0,
            'dbscan_threshold_athr': 0.0,
            'dbscan_clustering_algorithm': 1,
            'bounding_box_scale': 35.0,
            'bounding_box_offset': 0.5,
        }]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/image')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8'],
        }],
        remappings=[('buffer/input0', 'data_loader/image'),
                    ('input0', 'image')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_detection2_d_array',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'detectnet/detections')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TesetIsaacROSDetectNet.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            encoder_node,
            triton_node,
            detectnet_decoder_node
        ],
        output='screen',
    )

    return [composable_node_container]

def generate_test_description():
    MODELS_ROOT = os.path.join(TesetIsaacROSDetectNet.get_assets_root_path(), 'models')
    if not os.path.exists(os.path.dirname(ENGINE_FILE_PATH)):
        os.makedirs(os.path.dirname(ENGINE_FILE_PATH))
    shutil.copy(
        os.path.join(MODELS_ROOT, MODEL_CONFIG_FILE_NAME),
        ENGINE_FILE_DIR)

    # Generate engine file using tao-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        tao_converter_args = [
            '-k', 'tlt_encode',
            '-d', '3,544,960',
            '-p', 'input_1,1x3x544x960,1x3x544x960,1x3x544x960',
            '-t', 'int8',
            '-c', f'{MODELS_ROOT}/{MODEL_NAME}/resnet34_peoplenet_int8.txt',
            '-e', ENGINE_FILE_PATH,
            '-o', 'output_cov/Sigmoid,output_bbox/BiasAdd',
            f'{MODELS_ROOT}/{MODEL_NAME}/resnet34_peoplenet_int8.etlt'
        ]
        TaoConverter()(tao_converter_args)
    return TesetIsaacROSDetectNet.generate_test_description_with_nsys(launch_setup)


class TesetIsaacROSDetectNet(ROS2BenchmarkTest):
    """Performance test for Isaac ROS DetectNet graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS DetectNet Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    # Amount of seconds to wait for DetectNet to be initialized
    DETECTNET_WAIT_SEC = 60

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for DetectNet Node to be launched
        time.sleep(self.DETECTNET_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
