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

import os

from isaac_ros_test import IsaacROSBaseTest
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark.utils.ros2_utility import ClientUtility
from ros2_benchmark_interfaces.srv import PlayMessages, StartRecording


def generate_test_description():
    """Initialize test nodes and generate test description."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    rosbag_path = os.path.join(dir_path, 'pol.bag')

    playback_node = ComposableNode(
        name='NitrosPlaybackNode',
        namespace=NitrosPlaybackNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': ['nitros_image_bgr8'],
        }],
        remappings=[('buffer/input0', '/buffer/image'),
                    ('input0', '/image')],
    )

    playback_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='playback_container',
        namespace=NitrosPlaybackNode.generate_namespace(),
        executable='component_container_mt',
        composable_node_descriptions=[playback_node],
        output='screen'
    )

    # Play rosbag for the playback node to record messages
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--remap', 'image:=/buffer/image'],
        output='screen')

    return NitrosPlaybackNode.generate_test_description([rosbag_play, playback_container])


class NitrosPlaybackNode(IsaacROSBaseTest):
    """An unit test class for NitrosPlaybackNode."""

    def test_playback_node_services(self):
        """Test services hosted in NitrosPlaybackNode."""
        SERVICE_SETUP_TIMEOUT_SEC = 5
        SERVICE_TIMEOUT_SEC = 20
        SERVICE_FUTURE_TIMEOUT_SEC = 25

        # Create a start_recording service client
        start_recording_client = ClientUtility.create_service_client_blocking(
            self.node, StartRecording, 'start_recording', SERVICE_SETUP_TIMEOUT_SEC)
        self.assertIsNotNone(start_recording_client)

        # Create a play_messages service client
        play_messages_client = ClientUtility.create_service_client_blocking(
            self.node, PlayMessages, 'play_messages', SERVICE_SETUP_TIMEOUT_SEC)
        self.assertIsNotNone(play_messages_client)

        # Send a request to the start_recording service
        start_recording_request = StartRecording.Request()
        start_recording_request.buffer_length = 10
        start_recording_request.timeout = SERVICE_TIMEOUT_SEC
        start_recording_future = start_recording_client.call_async(start_recording_request)

        # Wait for the response from the start_recording service
        start_recording_response = ClientUtility.get_service_response_from_future_blocking(
            self.node, start_recording_future, SERVICE_FUTURE_TIMEOUT_SEC)
        self.assertIsNotNone(start_recording_response)
        self.node.get_logger().info('Received response from the start_recording service:')
        self.node.get_logger().info(str(start_recording_response))

        # Send a request to the play_messages service
        play_messages_request = PlayMessages.Request()
        play_messages_request.target_publisher_rate = 30.0
        play_messages_future = play_messages_client.call_async(play_messages_request)

        # Wait for the response from the play_messages service
        play_messages_response = ClientUtility.get_service_response_from_future_blocking(
            self.node, play_messages_future, SERVICE_FUTURE_TIMEOUT_SEC)
        self.assertIsNotNone(play_messages_response)
        self.node.get_logger().info('Received response from the play_messages service:')
        self.node.get_logger().info(str(play_messages_response))
