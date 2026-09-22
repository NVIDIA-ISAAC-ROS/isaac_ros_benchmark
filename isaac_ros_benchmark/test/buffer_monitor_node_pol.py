# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import time

from isaac_ros_test import IsaacROSBaseTest
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ros2_benchmark.utils.ros2_utility import ClientUtility
from ros2_benchmark_interfaces.srv import StartMonitoring, StopMonitoring


def generate_test_description():
    """Launch the buffer monitor node and its input rosbag."""
    rosbag_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'pol.bag')
    monitor = ComposableNode(
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferMonitorNode',
        name='MonitorNode0',
        namespace=BufferMonitorNodeTest.generate_namespace(),
        parameters=[{
            'monitor_index': 0,
            'monitor_data_format': 'sensor_msgs/msg/Image',
        }],
        remappings=[('output', '/image')],
    )
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container',
        name='monitor_container',
        namespace=BufferMonitorNodeTest.generate_namespace(),
        composable_node_descriptions=[monitor],
        output='screen',
    )
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop'],
        output='screen',
    )
    return BufferMonitorNodeTest.generate_test_description([rosbag_play, container])


class BufferMonitorNodeTest(IsaacROSBaseTest):
    """Validate monitoring of typed Image messages."""

    def test_monitor_node_services(self):
        """Record a monitored Image timestamp."""
        start_client = ClientUtility.create_service_client_blocking(
            self.node, StartMonitoring, 'monitor_node0_start_monitoring', 5)
        stop_client = ClientUtility.create_service_client_blocking(
            self.node, StopMonitoring, 'monitor_node0_stop_monitoring', 5)
        self.assertIsNotNone(start_client)
        self.assertIsNotNone(stop_client)

        start_request = StartMonitoring.Request()
        start_request.message_count = 1
        start_future = start_client.call_async(start_request)
        self.assertIsNotNone(
            ClientUtility.get_service_response_from_future_blocking(
                self.node, start_future, 25))

        time.sleep(1)

        stop_future = stop_client.call_async(StopMonitoring.Request())
        stop_response = ClientUtility.get_service_response_from_future_blocking(
            self.node, stop_future, 25)
        self.assertIsNotNone(stop_response)
        self.assertGreater(len(stop_response.end_timestamps.keys), 0)
