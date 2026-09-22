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

from isaac_ros_test import IsaacROSBaseTest
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from ros2_benchmark.utils.ros2_utility import ClientUtility
from ros2_benchmark_interfaces.srv import PlayMessages, StartRecording


def generate_test_description():
    """Launch the buffer playback node and its input rosbag."""
    rosbag_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'pol.bag')
    playback = ComposableNode(
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::BufferPlaybackNode',
        name='BufferPlaybackNode',
        namespace=BufferPlaybackNodeTest.generate_namespace(),
        parameters=[{
            'data_formats': ['sensor_msgs/msg/Image', 'sensor_msgs/msg/CameraInfo'],
        }],
        remappings=[
            ('buffer/input0', '/buffer/image'),
            ('input0', '/image'),
            ('buffer/input1', '/buffer/camera_info'),
            ('input1', '/camera_info'),
        ],
    )
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        name='playback_container',
        namespace=BufferPlaybackNodeTest.generate_namespace(),
        composable_node_descriptions=[playback],
        output='screen',
    )
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop',
             '--remap', 'image:=/buffer/image',
             'camera_info:=/buffer/camera_info'],
        output='screen',
    )
    return BufferPlaybackNodeTest.generate_test_description([rosbag_play, container])


class BufferPlaybackNodeTest(IsaacROSBaseTest):
    """Validate buffer playback recording and playback services."""

    def test_playback_node_services(self):
        """Record and replay CUDA-backed Image messages."""
        start_client = ClientUtility.create_service_client_blocking(
            self.node, StartRecording, 'start_recording', 5)
        play_client = ClientUtility.create_service_client_blocking(
            self.node, PlayMessages, 'play_messages', 5)
        self.assertIsNotNone(start_client)
        self.assertIsNotNone(play_client)

        start_request = StartRecording.Request()
        start_request.buffer_length = 10
        start_request.timeout = 20
        start_future = start_client.call_async(start_request)
        start_response = ClientUtility.get_service_response_from_future_blocking(
            self.node, start_future, 25)
        self.assertIsNotNone(start_response)
        self.assertTrue(start_response.success)
        self.assertGreaterEqual(start_response.recorded_message_count, 20)
        self.assertEqual(len(start_response.recorded_topic_message_counts), 2)
        for topic_count in start_response.recorded_topic_message_counts:
            self.assertGreaterEqual(topic_count.message_count, 10)

        play_request = PlayMessages.Request()
        play_request.target_publisher_rate = 30.0
        play_future = play_client.call_async(play_request)
        play_response = ClientUtility.get_service_response_from_future_blocking(
            self.node, play_future, 25)
        self.assertIsNotNone(play_response)
        self.assertTrue(play_response.success)
        self.assertGreater(len(play_response.timestamps.keys), 0)
