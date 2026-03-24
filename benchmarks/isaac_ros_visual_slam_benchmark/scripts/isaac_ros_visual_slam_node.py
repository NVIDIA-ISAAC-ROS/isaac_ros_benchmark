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
Performance test for the VisualSlam node.

The graph consists of the following:
- Preprocessors:
    None
- Graph under Test:
    1. VisualSlamNode: performs stereo visual simultaneous localization and mapping

Required:
- Packages:
    - isaac_ros_visual_slam
    - hawk_description
- Datasets:
    - assets/datasets/r2b_dataset/r2b_cafe
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_cafe'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for VSLAM node."""
    hawk_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hawk_description'),
                'launch',
                'hawk_description.launch.py',
            )
        )
    )

    visual_slam_node = ComposableNode(
        name='VisualSlamNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        remappings=[
            ('visual_slam/image_0', 'left/image_raw'),
            ('visual_slam/camera_info_0', 'left/camera_info'),
            ('visual_slam/image_1', 'right/image_raw'),
            ('visual_slam/camera_info_1', 'right/camera_info')],
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': False,
            'base_frame': 'base_link',
            'camera_optical_frames': [
                 'hawk_stereo_camera_left_optical',
                 'hawk_stereo_camera_right_optical',
            ],
        }],
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('hawk_0_left_rgb_image', 'buffer/image_left'),
            ('hawk_0_left_rgb_camera_info', 'buffer/camera_info_left'),
            ('hawk_0_right_rgb_image', 'buffer/image_right'),
            ('hawk_0_right_rgb_camera_info', 'buffer/camera_info_right')]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'sensor_msgs/msg/Image',
                'sensor_msgs/msg/CameraInfo',
                'sensor_msgs/msg/Image',
                'sensor_msgs/msg/CameraInfo'],
        }],
        remappings=[('buffer/input0', 'buffer/image_left'),
                    ('input0', 'left/image_raw'),
                    ('buffer/input1', 'buffer/camera_info_left'),
                    ('input1', 'left/camera_info'),
                    ('buffer/input2', 'buffer/image_right'),
                    ('input2', 'right/image_raw'),
                    ('buffer/input3', 'buffer/camera_info_right'),
                    ('input3', 'right/camera_info')],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nav_msgs/msg/Odometry',
        }],
        remappings=[
            ('output', 'visual_slam/tracking/odometry')],
    )

    composable_node_container = ComposableNodeContainer(
        name='vslam_container',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            playback_node,
            monitor_node,
            visual_slam_node
        ],
        output='screen'
    )

    return [hawk_description_launch, composable_node_container]


def generate_test_description():
    return TestIsaacROSVisualSlamNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSVisualSlamNode(ROS2BenchmarkTest):
    """Performance test for the VisualSlam node."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS VisualSlamNode Benchmark',
        benchmark_mode=BenchmarkMode.TIMELINE,
        input_data_path=ROSBAG_PATH,
        # The number of frames to be buffered (0 means not restricted)
        playback_message_buffer_size=0,
        # Publish /tf_static messages beforehand
        publish_tf_static_messages_in_set_data=True,
        start_recording_service_timeout_sec=60,
        start_recording_service_future_timeout_sec=65,
        default_service_future_timeout_sec=75
    )

    def test_benchmark(self):
        self.run_benchmark()
