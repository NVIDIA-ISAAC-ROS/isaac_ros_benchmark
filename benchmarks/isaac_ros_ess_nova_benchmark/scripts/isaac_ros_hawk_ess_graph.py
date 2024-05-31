# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Live benchmarking Isaac ROS Hawk ESS graph.

The graph consists of the following:
- Graph under Test:
    1. CorrelatedTimestampDriver: timestamp correaltor
    2. HawkNode: publishes images
    3. LeftRectifyNode, RightRectifyNode: rectifies images
    4. LeftResizeNode, RightResizeNode: resizes images to 960 x 576
    5. ESSDisparityNode: creates disparity images from stereo pair
    6. DisparityToDepthNode: converts disparity to depth
    7. DepthToPointCloudNode: converts depth to pointcloud

Required:
- Packages:
    - isaac_ros_correlated_timestamp_driver
    - isaac_ros_hawk
    - isaac_ros_image_proc
    - isaac_ros_ess
    - isaac_ros_stereo_image_proc
    - isaac_ros_depth_image_proc
- Models:
    - assets/models/ess/ess.etlt
"""

import os
import time

from isaac_ros_ess.engine_generator import ESSEngineGenerator

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BasicPerformanceCalculator, BenchmarkMode
from ros2_benchmark import ImageResolution
from ros2_benchmark import MonitorPerformanceCalculatorsInfo
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

MODEL_FILE_NAME = 'ess/ess.etlt'
ENGINE_FILE_PATH = 'ess/ess.engine'
NETWORK_WIDTH = 960
NETWORK_HEIGHT = 576
IMAGE_RESOLUTION = ImageResolution.HD


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS Hawk ESS graph."""
    MODELS_ROOT = os.path.join(TestIsaacROSHawkESSGraph.get_assets_root_path(), 'models')
    MODEL_ENGINE_PATH = os.path.join(MODELS_ROOT, ENGINE_FILE_PATH)
    correlated_timestamp_driver_node = ComposableNode(
        name='CorrelatedTimestampDriver',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_correlated_timestamp_driver',
        plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
        parameters=[{'use_time_since_epoch': False,
                     'nvpps_dev_name': '/dev/nvpps0'}])

    hawk_node = ComposableNode(
        name='HawkNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{
            'module_id': 5,
        }]
    )

    left_rectify_node = ComposableNode(
        name='LeftRectifyNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='RightRectifyNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    left_resize_node = ComposableNode(
        name='LeftResizeNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
                'output_width': NETWORK_WIDTH,
                'output_height': NETWORK_HEIGHT,
                'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'left/camera_info_rect'),
            ('image', 'left/image_rect'),
            ('resize/camera_info', 'left/camera_info_resize'),
            ('resize/image', 'left/image_resize')]
    )

    right_resize_node = ComposableNode(
        name='RightResizeNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
                'output_width': NETWORK_WIDTH,
                'output_height': NETWORK_HEIGHT,
                'keep_aspect_ratio': True
        }],
        remappings=[
            ('camera_info', 'right/camera_info_rect'),
            ('image', 'right/image_rect'),
            ('resize/camera_info', 'right/camera_info_resize'),
            ('resize/image', 'right/image_resize')]
    )

    disparity_node = ComposableNode(
        name='ESSDisparityNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{'engine_file_path': MODEL_ENGINE_PATH}],
        remappings=[
            ('left/camera_info', 'left/camera_info_resize'),
            ('left/image_rect', 'left/image_resize'),
            ('right/camera_info', 'right/camera_info_resize'),
            ('right/image_rect', 'right/image_resize')
        ]
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
    )

    depth_to_pointcloud_node = ComposableNode(
        name='DepthToPointCloudNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
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
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'left/image_raw')]
    )

    right_image_monitor_node = ComposableNode(
        name='RightImageMonitorNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 1,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'right/image_raw')]
    )

    disparity_monitor_node = ComposableNode(
        name='DisparityMonitorNode',
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
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
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
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
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
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
        namespace=TestIsaacROSHawkESSGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            correlated_timestamp_driver_node,
            hawk_node,
            left_rectify_node,
            right_rectify_node,
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
    MODELS_ROOT = os.path.join(TestIsaacROSHawkESSGraph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    # Generate engine file using tao-converter
    if not os.path.isfile(os.path.join(MODELS_ROOT, ENGINE_FILE_PATH)):
        gen = ESSEngineGenerator(etlt_model=MODEL_FILE_PATH)
        gen.generate()
    return TestIsaacROSHawkESSGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSHawkESSGraph(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS Hawk ESS Graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Hawk ESS Graph Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': f'({NETWORK_WIDTH},{NETWORK_HEIGHT})'},
        monitor_info_list=[
            MonitorPerformanceCalculatorsInfo(
                'monitor_node0',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Left Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node1',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Right Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node2',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Disparity',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node3',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Depth',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'monitor_node4',
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
        MODELS_ROOT = os.path.join(TestIsaacROSHawkESSGraph.get_assets_root_path(), 'models')
        while not os.path.isfile(os.path.join(MODELS_ROOT, ENGINE_FILE_PATH)):
            time.sleep(1)
        # Wait for ESS Node to be launched
        self.get_logger().info(
            f'Detected engine files, waiting {self.ESS_WAIT_SEC} secs before benchmark begins...')
        time.sleep(self.ESS_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
