# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Live benchmarking Isaac ROS Realsense ESS graph.

The graph consists of the following:
- Graph under Test:
    1. Realsense2Camera: publishes images
    2. LeftFormatConverterNode, RightFormatConverterNode: converts images to rgb8
    3. LeftResizeNode, RightResizeNode: resizes images to 960 x 576
    4. ESSDisparityNode: creates disparity images from stereo pair
    5. DisparityToDepthNode: converts disparity to depth
    6. DepthToPointCloudNode: converts depth to pointcloud

Required:
- Packages:
    - realsense2_camera
    - isaac_ros_image_proc
    - isaac_ros_ess
    - isaac_ros_stereo_image_proc
    - isaac_ros_depth_image_proc
- Models:
    - assets/models/ess/ess.etlt
"""

import os
import time

from isaac_ros_benchmark import TaoConverter

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import BasicPerformanceCalculator, BenchmarkMode
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest
from ros2_benchmark import MonitorPerformanceCalculatorsInfo

MODEL_FILE_NAME = 'ess/ess.etlt'
ENGINE_FILE_PATH = '/tmp/ess.engine'
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 576
IMAGE_RESOLUTION = ImageResolution.VGA

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS Realsense ESS graph."""
    # RealSense
    realsense_config_file_path = os.path.join(
        TestIsaacROSRealsenseESSGraph.get_assets_root_path(),
        'configs', 'realsense.yaml')

    realsense_node = ComposableNode(
        name='Realsense2Camera',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config_file_path],
        remappings=[
            ('infra1/image_rect_raw', 'left/image_rect_raw_mono'),
            ('infra2/image_rect_raw', 'right/image_rect_raw_mono'),
            ('infra1/camera_info', 'left/camerainfo'),
            ('infra2/camera_info', 'right/camerainfo')
        ]
    )

    left_image_format_converter_node = ComposableNode(
        name='LeftFormatConverterNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'left/image_rect_raw_mono'),
            ('image', 'left/image_rect_raw')]
    )

    right_image_format_converter_node = ComposableNode(
        name='RightFormatConverterNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_rect_raw_mono'),
            ('image', 'right/image_rect_raw')]
    )

    left_resize_node = ComposableNode(
        name='LeftResizeNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
                'output_width': NETWORK_WIDTH,
                'output_height': NETWORK_HEIGHT,
                'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'left/camerainfo'),
            ('image', 'left/image_rect_raw'),
            ('resize/camera_info', 'left/camera_info_resize'),
            ('resize/image', 'left/image_resize')]
    )

    right_resize_node = ComposableNode(
        name='RightResizeNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
                'output_width': NETWORK_WIDTH,
                'output_height': NETWORK_HEIGHT,
                'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'right/camerainfo'),
            ('image', 'right/image_rect_raw'),
            ('resize/camera_info', 'right/camera_info_resize'),
            ('resize/image', 'right/image_resize')]
    )

    disparity_node = ComposableNode(
        name='ESSDisparityNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': ENGINE_FILE_PATH}],
        remappings=[
            ('left/camera_info', 'left/camera_info_resize'),
            ('left/image_rect', 'left/image_resize'),
            ('right/camera_info', 'right/camera_info_resize'),
            ('right/image_rect', 'right/image_resize')
        ]
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
    )

    depth_to_pointcloud_node = ComposableNode(
        name='DepthToPointCloudNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::PointCloudXyzNode',
        parameters=[{
                'skip': 21
        }],
        remappings=[
            ('image_rect', 'depth'),
            ('camera_info', 'left/camera_info_resize'),
            ('points', 'ess_points')
        ]
    )

    left_image_monitor_node = ComposableNode(
        name='LeftImageMonitorNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'sensor_msgs/msg/Image',
            'use_nitros_type_monitor_sub': False,
        }],
        remappings=[('output', 'left/image_rect_raw_mono')]
    )

    right_image_monitor_node = ComposableNode(
        name='RightImageMonitorNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 1,
            'monitor_data_format': 'sensor_msgs/msg/Image',
            'use_nitros_type_monitor_sub': False,
        }],
        remappings=[('output', 'right/image_rect_raw_mono')]
    )

    disparity_monitor_node = ComposableNode(
        name='DisparityMonitorNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 2,
            'monitor_data_format': 'nitros_disparity_image_32FC1',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'disparity')],
    )

    depth_monitor_node = ComposableNode(
        name='DepthMonitorNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 3,
            'monitor_data_format': 'nitros_image_32FC1',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'depth')],
    )

    points_monitor_node = ComposableNode(
        name='PointCloudMonitorNode',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 4,
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'ess_points')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSRealsenseESSGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            realsense_node,
            left_image_format_converter_node,
            right_image_format_converter_node,
            left_resize_node,
            right_resize_node,
            disparity_node,
            disparity_to_depth_node,
            depth_to_pointcloud_node,
            left_image_monitor_node,
            right_image_monitor_node,
            disparity_monitor_node,
            depth_monitor_node,
            points_monitor_node
        ],
        output='screen'
    )

    return [composable_node_container]

def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSRealsenseESSGraph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using tao-converter
    if not os.path.isfile(ENGINE_FILE_PATH):
        tao_converter_args = [
            '-k', 'ess',
            '-t', 'fp16',
            '-e', ENGINE_FILE_PATH,
            '-o', 'output_left,output_conf', MODEL_FILE_PATH
        ]
        TaoConverter()(tao_converter_args)

    return TestIsaacROSRealsenseESSGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSRealsenseESSGraph(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS Realsense ESS Graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Realsense ESS Graph Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': f'({NETWORK_WIDTH},{NETWORK_HEIGHT})'},
        monitor_info_list=[
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring0',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Realsense Left Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring1',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Realsense Right Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring2',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Disparity',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring3',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Depth',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring4',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Point Cloud',
                    'message_key_match': True
                })])
        ]
    )

    # Amount of seconds to wait for Triton Engine to be initialized
    ESS_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(ENGINE_FILE_PATH):
            time.sleep(1)
        # Wait for ESS Node to be launched
        time.sleep(self.ESS_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
