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
Performance test for the Isaac ROS NITROS Bridge.

The graph consists of the following:
- Graph under Test:
    1. NitrosBridge

Required:
- Packages:
    - isaac_ros_nitros_bridge_ros2
    - isaac_ros_nitros_bridge_ros1
    - isaac_ros_ros1_forward
    - isaac_ros_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_mezzanine
"""

import os
import subprocess
import time

from launch_ros.actions import ComposableNodeContainer
from launch.actions import ExecuteProcess
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.FULL_HD
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_mezzanine'
BRIDGE_CONFIG_PATH = 'src/isaac_ros_nitros_bridge/config/nitros_bridge.yaml'
ROS1_INSTALL_PATH = 'install_isolated/setup.bash'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS NITROS bridge."""

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info')]
    )

    prep_resize_node = ComposableNode(
        name='PrepResizeNode',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/left_image'),
                    ('camera_info', 'data_loader/left_camera_info'),
                    ('resize/image', 'buffer/image_raw'),
                    ('resize/camera_info', 'buffer/camera_info')],
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8'],
        }],
        remappings=[('buffer/input0', 'buffer/image_raw'),
                    ('input0', 'ros2_input_image')]
    )

    ros2_converter = ComposableNode(
        name='ros2_converter',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='isaac_ros_nitros_bridge_ros2',
        plugin='nvidia::isaac_ros::nitros_bridge::ImageConverterNode',
        remappings=[
            ('ros2_output_bridge_image', 'ros1_input_bridge_image'),
            ('ros2_input_bridge_image', 'ros1_output_bridge_image')
        ]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_image_rgb8',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'ros2_output_image')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacNitrosBridgeNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_resize_node,
            playback_node,
            ros2_converter,
            monitor_node,
        ],
        output='screen'
    )

    return [composable_node_container]

def generate_test_description():
    return TestIsaacNitrosBridgeNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacNitrosBridgeNode(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Nitros Bridge."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Nitros Bridge Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        play_messages_service_future_timeout_sec = 60.0
    )

    def pre_benchmark_hook(self):
        roscore_cmd = 'source /opt/ros/noetic/setup.bash && roscore'
        self.roscore_proc = subprocess.Popen(
            roscore_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/bash"
        )
        # Wait one second to ensure roscore is fully launched
        time.sleep(1)

        ros1_setup_path = os.path.join(self.get_ros1_ws_path(), ROS1_INSTALL_PATH)
        stop_ros_nodes_cmd = f'source {ros1_setup_path} && rosnode kill -a'
        self.stop_ros_nodes_proc = subprocess.run(
            stop_ros_nodes_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/bash"
        )

        bridge_config_absolute_path = os.path.join(self.get_ros1_ws_path(), BRIDGE_CONFIG_PATH)
        ros1_converter_cmd = f'source {ros1_setup_path} \
            && rosparam load {bridge_config_absolute_path} \
            && roslaunch isaac_ros_nitros_bridge_ros1 r2b_converter.launch'
        self.ros1_converter_proc = subprocess.Popen(
            ros1_converter_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/bash"
        )

        ros1_bridge_cmd = 'ros2 run ros1_bridge parameter_bridge'
        self.ros1_bridge_proc = subprocess.Popen(
            ros1_bridge_cmd,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

    def test_benchmark(self):
        self.run_benchmark()
