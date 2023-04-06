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
Performance test for the Isaac ROS FreespaceSegmentationNode.

The graph consists of the following:
- Preprocessors:
    1. PrepLeftResizeNode, PrepRightResizeNode: resizes images to 960 x 576
    2. Bi3DNode: creates proximity segmentation disparity image from stereo pair
- Graph under Test:
    1. FreespaceSegmentationNode: creates occupancy grid from proximity segmentation disparity image

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_bi3d
    - isaac_ros_bi3d_freespace
- Datasets:
    - assets/datasets/r2b_dataset/r2b_lounge
- Models:
    - assets/models/bi3d/featnet.onnx
    - assets/models/bi3d/segnet.onnx
"""

import os
import time

from isaac_ros_benchmark import TRTConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import Resolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = Resolution(960, 576)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_lounge'
FEATNET_MODEL_FILE_NAME = 'bi3d/featnet.onnx'
SEGNET_MODEL_FILE_NAME = 'bi3d/segnet.onnx'
FEATNET_ENGINE_FILE_PATH = '/tmp/featnet.engine'
SEGNET_ENGINE_FILE_PATH = '/tmp/segnet.engine'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS FreespaceSegmentationNode."""

    freespace_node = ComposableNode(
        name='FreespaceSegmentationNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_bi3d_freespace',
        plugin='nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode',
        parameters=[{
            'base_link_frame': 'base_link',
            'camera_frame': 'camera',
            'f_x': 2000.0,
            'f_y': 2000.0,
            'grid_width': 200,
            'grid_height': 100,
            'grid_resolution': 0.01
        }]
    )

    tf_publisher = ComposableNode(
        name='StaticTransformBroadcasterNode',
        package='tf2_ros',
        plugin='tf2_ros::StaticTransformBroadcasterNode',
        parameters=[{
            'frame_id': 'base_link',
            'child_frame_id': 'camera',
            'translation.x': -0.3,
            'translation.y': 0.2,
            'translation.z': 0.5,
            'rotation.x': -0.12,
            'rotation.y': 0.98,
            'rotation.z': -0.17,
            'rotation.w': 0.02,
        }]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
                    ('hawk_0_right_rgb_image', 'data_loader/right_image'),
                    ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')]
    )

    prep_left_resize_node = ComposableNode(
        name='PrepLeftResizeNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/left_image'),
                    ('camera_info', 'data_loader/left_camera_info'),
                    ('resize/image', 'left_image_bi3d'),
                    ('resize/camera_info', 'left/camera_info')],
    )

    prep_right_resize_node = ComposableNode(
        name='PrepRightResizeNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/right_image'),
                    ('camera_info', 'data_loader/right_camera_info'),
                    ('resize/image', 'right_image_bi3d'),
                    ('resize/camera_info', 'right/camera_info')],
    )

    bi3d_node = ComposableNode(
        name='Bi3DNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
            'featnet_engine_file_path': FEATNET_ENGINE_FILE_PATH,
            'segnet_engine_file_path': SEGNET_ENGINE_FILE_PATH,
            'max_disparity_values': 3,
            'disparity_values': [10, 18, 26]
        }],
        remappings=[('bi3d_node/bi3d_output', 'buffer/disparity_image')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_disparity_image_32FC1'],
        }],
        remappings=[('buffer/input0', 'buffer/disparity_image'),
                    ('input0', 'bi3d_mask')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_occupancy_grid',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'freespace_segmentation/occupancy_grid')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSBi3DFreespace.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_left_resize_node,
            prep_right_resize_node,
            bi3d_node,
            playback_node,
            monitor_node,
            freespace_node,
            tf_publisher
        ],
        output='screen'
    )

    return [composable_node_container]

def generate_test_description():
    MODELS_ROOT = os.path.join(
        TestIsaacROSBi3DFreespace.get_assets_root_path(), 'models')
    FEATNET_MODEL_FILE_PATH = os.path.join(
        MODELS_ROOT, FEATNET_MODEL_FILE_NAME)
    SEGNET_MODEL_FILE_PATH = os.path.join(MODELS_ROOT, SEGNET_MODEL_FILE_NAME)

    # Generate engine file using trt-converter
    def generate_engine(use_dla_core):
        if not os.path.isfile(FEATNET_ENGINE_FILE_PATH):
            trt_converter_args = [
                f'--onnx={FEATNET_MODEL_FILE_PATH}',
                f'--saveEngine={FEATNET_ENGINE_FILE_PATH}',
                '--int8'
            ]
            if use_dla_core:
                trt_converter_args.append('--useDLACore=0')
                trt_converter_args.append('--allowGPUFallback')
            TRTConverter()(trt_converter_args)
        if not os.path.isfile(SEGNET_ENGINE_FILE_PATH):
            trt_converter_args = [
                f'--onnx={SEGNET_MODEL_FILE_PATH}',
                f'--saveEngine={SEGNET_ENGINE_FILE_PATH}',
                '--int8',
            ]
            if use_dla_core:
                trt_converter_args.append('--useDLACore=0')
                trt_converter_args.append('--allowGPUFallback')
            TRTConverter()(trt_converter_args)

    try:
        generate_engine(True)
    except Exception as e:
        if 'Cannot create DLA engine' in str(e):
            print('DLA engine is not supported')
            print('Run TRT converter again without using DLA engine')
            generate_engine(False)
        else:
            raise e

    return TestIsaacROSBi3DFreespace.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSBi3DFreespace(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS FreespaceSegmentationNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Bi3D FreespaceSegmentationNode Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=3500.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        linear_scan_acceptable_frame_loss_fraction=0.05,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    # Amount of time to wait for trtexec to compile engine files
    TRT_WAIT_SEC = 500
    # Amount of seconds to wait for Triton Engine to be initialized
    BI3D_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        end_time = time.time() + self.TRT_WAIT_SEC
        while time.time() < end_time:
            if os.path.isfile(FEATNET_ENGINE_FILE_PATH) and \
               os.path.isfile(SEGNET_ENGINE_FILE_PATH):
                break
        self.assertTrue(os.path.isfile(FEATNET_ENGINE_FILE_PATH),
                        'Featnet engine file was not generated in time.')
        self.assertTrue(os.path.isfile(SEGNET_ENGINE_FILE_PATH),
                        'Segnet engine file was not generated in time.')
        # Wait for Bi3D Node to be launched
        time.sleep(self.BI3D_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
