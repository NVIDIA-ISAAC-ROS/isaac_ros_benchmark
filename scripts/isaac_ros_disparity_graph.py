# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for the Isaac ROS stereo image graph.

The graph consists of the following:
- Preprocessors:
    1. PrepLeftResizeNode, PrepRightResizeNode: resizes images to quarter HD
- Graph under Test:
    1. DisparityNode: creates disparity images from stereo pair
    2. PointCloudNode: converts disparity to pointcloud

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_stereo_image_proc
- Datasets:
    - assets/datasets/r2b_dataset/r2b_datacenter
"""

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.QUARTER_HD
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_datacenter'

def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS stereo image graph."""

    disparity_node = ComposableNode(
        name='DisparityNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
        parameters=[{
            'backends': 'CUDA',
            'max_disparity': 64.0,
        }])

    pointcloud_node = ComposableNode(
        name='PointCloudNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
        parameters=[{
                'use_color': False,
        }],
        remappings=[('left/image_rect_color', 'left/image_rect')])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[('hawk_0_left_rgb_image', 'data_loader/left_image'),
                    ('hawk_0_left_rgb_camera_info', 'data_loader/left_camera_info'),
                    ('hawk_0_right_rgb_image', 'data_loader/right_image'),
                    ('hawk_0_right_rgb_camera_info', 'data_loader/right_camera_info')]
    )

    prep_left_resize_node = ComposableNode(
        name='PrepLeftResizeNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/left_image'),
                    ('camera_info', 'data_loader/left_camera_info'),
                    ('resize/image', 'buffer/left/image_rect'),
                    ('resize/camera_info', 'buffer/left/camera_info')],
    )

    prep_right_resize_node = ComposableNode(
        name='PrepRightResizeNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
        }],
        remappings=[('image', 'data_loader/right_image'),
                    ('camera_info', 'data_loader/right_camera_info'),
                    ('resize/image', 'buffer/right/image_rect'),
                    ('resize/camera_info', 'buffer/right/camera_info')],
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_image_bgr8',
                'nitros_image_bgr8',
                'nitros_camera_info',
                'nitros_camera_info'
            ],
        }],
        remappings=[('buffer/input0', 'buffer/left/image_rect'),
                    ('input0', 'left/image_rect'),
                    ('buffer/input1', 'buffer/right/image_rect'),
                    ('input1', 'right/image_rect'),
                    ('buffer/input2', 'buffer/left/camera_info'),
                    ('input2', 'left/camera_info'),
                    ('buffer/input3', 'buffer/right/camera_info'),
                    ('input3', 'right/camera_info')]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_point_cloud',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'points2')],
    )

    composable_node_container = ComposableNodeContainer(
        name='stereo_graph_container',
        namespace=TestIsaacROSStereoGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_left_resize_node,
            prep_right_resize_node,
            playback_node,
            monitor_node,
            disparity_node,
            pointcloud_node
        ],
        output='screen',
    )

    return [composable_node_container]

def generate_test_description():
    return TestIsaacROSStereoGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSStereoGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS disparity image graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS (DisparityNode) Stereo Image Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=1000.0,
        publisher_lower_frequency=10.0,
        # The number of frames to be buffered
        playback_message_buffer_size=10,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION}
    )

    def test_benchmark(self):
        self.run_benchmark()
