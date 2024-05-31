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
- Datasets:
    - assets/datasets/r2b_dataset/r2b_cafe
"""

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_cafe'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for VSLAM node."""
    # We add a static tf here because the the one from the bag is wrong.
    static_tf_left = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', '1', 'D455_1', 'D455_1:left_ir_corrected'],
        output='screen',
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
            ('visual_slam/camera_info_1', 'right/camera_info'),
            ('visual_slam/imu', 'camera/imu')],
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'base_frame': 'base_link',
            'camera_optical_frames': [
                 'D455_1:left_ir_corrected',
                 'D455_1:right_ir',
            ],
        }],
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('d455_1_left_ir_image', 'buffer/image_left'),
            ('d455_1_left_ir_camera_info', 'buffer/camera_info_left'),
            ('d455_1_right_ir_image', 'buffer/image_right'),
            ('d455_1_right_ir_camera_info', 'buffer/camera_info_right'),
            ('d455_1_imu', 'buffer/d455_1_imu')]
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
                'sensor_msgs/msg/CameraInfo',
                'sensor_msgs/msg/Imu'],
        }],
        remappings=[('buffer/input0', 'buffer/image_left'),
                    ('input0', 'left/image_raw'),
                    ('buffer/input1', 'buffer/camera_info_left'),
                    ('input1', 'left/camera_info'),
                    ('buffer/input2', 'buffer/image_right'),
                    ('input2', 'right/image_raw'),
                    ('buffer/input3', 'buffer/camera_info_right'),
                    ('input3', 'right/camera_info'),
                    ('buffer/input4', 'buffer/d455_1_imu'),
                    ('input4', 'camera/imu')],
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

    return [static_tf_left, composable_node_container]


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
