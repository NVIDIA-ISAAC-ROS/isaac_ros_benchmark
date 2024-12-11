# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Live performance test for the Isaac ROS data recorder (4 Hawk cameras)."""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_benchmark import NitrosMonitorUtility

import isaac_ros_hawk_nova_benchmark.hawk_benchmark_utility as hawk_benchmark_utility

from launch.actions import IncludeLaunchDescription, Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import LoadComposableNodes, Node

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest

NITROS_MONITOR_UTILITY = NitrosMonitorUtility()


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS data recorder (4 Hawk cameras)."""
    sensor_config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config/config_4_hawk.yaml')

    monitor_nodes = []
    benchmark_container = Node(
        name='benchmark_container',
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        on_exit=Shutdown(),
        output='both'
    )

    def add_hawk_monitor_nodes(node_namespace):
        monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
            NITROS_MONITOR_UTILITY,
            TestIsaacROSDataRecorder.generate_namespace(),
            node_namespace))
        monitor_nodes.append(NITROS_MONITOR_UTILITY.add_basic_perf_nitros_monitor(
            TestIsaacROSDataRecorder.generate_namespace(),
            node_namespace, 'nitros_compressed_image', 'left/image_compressed',
            f'({node_namespace}) Left Hawk Compressed Image',
            monitor_name=f'{node_namespace}_LeftHawkCompressedImageMonitorNode',
            message_key_match=True
        ))
        monitor_nodes.append(NITROS_MONITOR_UTILITY.add_basic_perf_nitros_monitor(
            TestIsaacROSDataRecorder.generate_namespace(),
            node_namespace, 'nitros_compressed_image', 'right/image_compressed',
            f'({node_namespace}) Right Hawk Compressed Image',
            monitor_name=f'{node_namespace}_RightHawkCompressedImageMonitorNode',
            message_key_match=True
        ))

    # Hawk raw and compressed image monitors
    add_hawk_monitor_nodes('front_stereo_camera')
    add_hawk_monitor_nodes('back_stereo_camera')
    add_hawk_monitor_nodes('left_stereo_camera')
    add_hawk_monitor_nodes('right_stereo_camera')

    load_benchmark_nodes = LoadComposableNodes(
        target_container='benchmark_container',
        composable_node_descriptions=monitor_nodes
    )

    isaac_ros_nova_recorder_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_nova_recorder'), 'launch')

    nova_recorder_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            isaac_ros_nova_recorder_launch_dir,
            '/nova_recorder.launch.py'
        ]),
        launch_arguments={
            'target_container': 'benchmark_container',
            'config': sensor_config_file,
            'headless': 'True',
        }.items(),
    )

    return [benchmark_container, load_benchmark_nodes, nova_recorder_launch]


def generate_test_description():
    return TestIsaacROSDataRecorder.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSDataRecorder(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS data recorder."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Data Recorder (4 Hawk Cameras) Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=5,
        test_iterations=5,
        collect_start_timestamps_from_monitors=True,
        pre_trial_run_wait_time_sec=20.0,
        monitor_info_list=[]
    )

    def test_benchmark(self):
        self.config.monitor_info_list = NITROS_MONITOR_UTILITY.get_monitor_info_list()
        self.run_benchmark()
