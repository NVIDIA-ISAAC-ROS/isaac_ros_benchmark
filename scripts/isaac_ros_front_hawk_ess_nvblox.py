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
Live benchmarking Isaac ROS front Hawk ESS Nvblox graph.

The graph consists of the following:
- Graph under Test:
    1. correlated_timestamp_driver_launch: timestamp correaltor
    2. nvblox_pipeline_launch: runs Hawk + ESS + Nvblox graph

Required:
- Packages:
    - carter_navigation
    - isaac_ros_correlated_timestamp_driver
    - isaac_ros_ess
    - isaac_ros_hawk
    - isaac_ros_image_proc
    - isaac_ros_stereo_image_proc
    - nvblox_ros
- Models:
    - assets/models/ess/ess.etlt
"""

import os
import time

from ament_index_python.packages import get_package_share_directory

from isaac_ros_benchmark import TaoConverter
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import BasicPerformanceCalculator, BenchmarkMode
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest
from ros2_benchmark import MonitorPerformanceCalculatorsInfo

MODEL_FILE_NAME = 'ess/ess.etlt'
ENGINE_FILE_PATH = '/tmp/ess.engine'
IMAGE_RESOLUTION = ImageResolution.HD

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS HawkNode."""
    carter_navigation_launch_include_dir = os.path.join(
        get_package_share_directory('carter_navigation'), 'launch', 'include')

    carter_container = Node(
        name='carter_container',
        package='rclcpp_components',
        executable='component_container_mt',
        sigterm_timeout=container_sigterm_timeout,
        output='screen')

    correlated_timestamp_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [carter_navigation_launch_include_dir, '/correlated_timestamp_driver.launch.py'])
    )

    nvblox_pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [carter_navigation_launch_include_dir, '/front_hawk_ess_nvblox.launch.py']),
        launch_arguments={
            'engine_file_path': 'turtlesim2'
        }.items()
    )

    left_image_monitor_node = ComposableNode(
        name='LeftImageMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', '/hawk_front/left/image_raw')]
    )

    right_image_monitor_node = ComposableNode(
        name='RightImageMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 1,
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', '/hawk_front/right/image_raw')]
    )

    ess_disparity_monitor_node = ComposableNode(
        name='ESSDisparityMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 2,
            'monitor_data_format': 'nitros_disparity_image_32FC1',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', '/hawk_front/disparity')]
    )

    ess_points_monitor_node = ComposableNode(
        name='ESSPointsMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 3,
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', '/hawk_front/ess_points')]
    )

    ess_points_monitor_node = ComposableNode(
        name='ESSPointsMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 3,
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', '/hawk_front/ess_points')]
    )

    nvblox_static_map_slice_monitor_node = ComposableNode(
        name='NvbloxStaticMapSliceMonitorNode',
        namespace=TestIsaacROSHawkESSNvbloxNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_index': 4,
            'monitor_data_format': 'nvblox_msgs/msg/DistanceMapSlice',
            'use_nitros_type_monitor_sub': False,
        }],
        remappings=[('output', '/hawk_front/nvblox_node/static_map_slice')]
    )

    load_monitor_nodes = LoadComposableNodes(
        target_container='carter_container',
        composable_node_descriptions=[
            left_image_monitor_node,
            right_image_monitor_node,
            ess_disparity_monitor_node,
            ess_points_monitor_node,
            nvblox_static_map_slice_monitor_node
        ]
    )

    return [carter_container,
            correlated_timestamp_driver_launch,
            nvblox_pipeline_launch,
            load_monitor_nodes]

def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSHawkESSNvbloxNode.get_assets_root_path(), 'models')
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

    return TestIsaacROSHawkESSNvbloxNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSHawkESSNvbloxNode(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS HawkNode."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Hawk ESS Nvblox Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION},
        monitor_info_list=[
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring0',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Left Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring1',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Hawk Right Image',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring2',
                [BasicPerformanceCalculator({
                    'report_prefix': 'ESS Disparity',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring3',
                [BasicPerformanceCalculator({
                    'report_prefix': 'ESS Point Cloud',
                    'message_key_match': True
                })]),
            MonitorPerformanceCalculatorsInfo(
                'start_monitoring4',
                [BasicPerformanceCalculator({
                    'report_prefix': 'Nvblox Static Map Slice',
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
