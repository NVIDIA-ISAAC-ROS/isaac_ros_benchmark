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

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_cafe'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for VSLAM node."""
    visual_slam_node = ComposableNode(
        name='VisualSlamNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='isaac_ros_visual_slam',
        plugin='isaac_ros::visual_slam::VisualSlamNode',
        remappings=[('stereo_camera/left/image', 'image_left'),
                    ('stereo_camera/left/camera_info', 'camera_info_left'),
                    ('stereo_camera/right/image', 'image_right'),
                    ('stereo_camera/right/camera_info', 'camera_info_right')],
        parameters=[{
                    'enable_rectified_pose': True,
                    'denoise_input_images': False,
                    'rectified_images': True
                    }],
        extra_arguments=[
            {'use_intra_process_comms': False}])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSVisualSlamNode.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('d455_1_left_ir_image', 'buffer/image_left'),
            ('d455_1_left_ir_camera_info', 'buffer/camera_info_left'),
            ('d455_1_right_ir_image', 'buffer/image_right'),
            ('d455_1_right_ir_camera_info', 'buffer/camera_info_right')]
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
                    ('input0', 'image_left'),
                    ('buffer/input1', 'buffer/camera_info_left'),
                    ('input1', 'camera_info_left'),
                    ('buffer/input2', 'buffer/image_right'),
                    ('input2', 'image_right'),
                    ('buffer/input3', 'buffer/camera_info_right'),
                    ('input3', 'camera_info_right')],
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

    return [composable_node_container]

def generate_test_description():
    return TestIsaacROSVisualSlamNode.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSVisualSlamNode(ROS2BenchmarkTest):
    """Performance test for the VisualSlam node."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS VisualSlamNode Benchmark',
        benchmark_mode=BenchmarkMode.SWEEPING,
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=500.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=100,
        # Fine tuned publisher frequency search parameters
        binary_search_acceptable_frame_rate_drop=15,
        linear_scan_acceptable_frame_rate_drop=10,
        start_recording_service_timeout_sec=60,
        start_recording_service_future_timeout_sec=65,
        start_monitoring_service_timeout_sec=60,
        default_service_future_timeout_sec=75
    )

    def test_benchmark(self):
        self.run_benchmark()
