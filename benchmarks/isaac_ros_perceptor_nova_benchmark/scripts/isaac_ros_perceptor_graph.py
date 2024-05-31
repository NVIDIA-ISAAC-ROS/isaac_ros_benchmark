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
"""Live benchmarking Isaac ROS Perceptor graph."""

import os

from ament_index_python.packages import get_package_prefix

from isaac_ros_benchmark import NitrosMonitorUtility

import isaac_ros_ess_benchmark.ess_benchmark_utility as ess_benchmark_utility
import isaac_ros_hawk_nova_benchmark.hawk_benchmark_utility as hawk_benchmark_utility

import isaac_ros_launch_utils as lu

from launch_ros.actions import LoadComposableNodes

from ros2_benchmark import BenchmarkMode, ROS2BenchmarkConfig, ROS2BenchmarkTest


NITROS_MONITOR_UTILITY = NitrosMonitorUtility()


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for live benchmarking Isaac ROS Perceptor graph."""
    shared_container_name = 'nova_container'

    # Run "perceptor.launch.py" configured for front, left, right hawks.
    perceptor_launch = lu.include(
        'nova_carter_bringup', 'launch/perceptor.launch.py',
        launch_arguments={
            'stereo_camera_configuration': 'front_left_right_configuration',
            'run_rviz': 'False',
            'run_foxglove': 'False',
        },
    )

    monitor_nodes = []
    ess_depth_monitor_topics = ['left/image_rect', 'right/image_rect', 'disparity', 'depth']
    # Front
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'front_stereo_camera'))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'front_stereo_camera',
        message_key_match=True,
        monitor_topics=ess_depth_monitor_topics))
    # Left
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'left_stereo_camera'))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'left_stereo_camera',
        message_key_match=True,
        monitor_topics=ess_depth_monitor_topics))
    # Right
    monitor_nodes.extend(hawk_benchmark_utility.create_hawk_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'right_stereo_camera'))
    monitor_nodes.extend(ess_benchmark_utility.create_ess_depth_graph_monitors(
        NITROS_MONITOR_UTILITY,
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'right_stereo_camera',
        message_key_match=True,
        monitor_topics=ess_depth_monitor_topics))

    # Cuvslam
    monitor_nodes.append(NITROS_MONITOR_UTILITY.add_basic_perf_nitros_monitor(
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'visual_slam', 'nav_msgs/msg/Odometry', 'tracking/odometry',
        'Visual odometry', 'OdometryMonitorNode',
        use_nitros_type_monitor_sub=False))

    # nvblox
    monitor_nodes.append(NITROS_MONITOR_UTILITY.add_basic_perf_nitros_monitor(
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'nvblox_node', 'nvblox_msgs/msg/Mesh', 'mesh',
        'Nvblox mesh', 'MeshMonitorNode',
        use_nitros_type_monitor_sub=False))
    monitor_nodes.append(NITROS_MONITOR_UTILITY.add_basic_perf_nitros_monitor(
        TestIsaacROSPerceptorGraph.generate_namespace(),
        'nvblox_node', 'nvblox_msgs/msg/DistanceMapSlice', 'static_map_slice',
        'Nvblox ESDF', 'EsdfMonitorNode',
        use_nitros_type_monitor_sub=False))

    load_monitor_nodes = LoadComposableNodes(
        target_container=shared_container_name,
        composable_node_descriptions=monitor_nodes
    )

    return [perceptor_launch, load_monitor_nodes]


def generate_test_description():
    print('Checking if ESS models are installed...')
    ess_install_script = os.path.join(
        get_package_prefix('isaac_ros_ess_models_install'),
        'lib/isaac_ros_ess_models_install/install_ess_models.sh')
    if os.system(ess_install_script) != 0:
        raise SystemExit(
            f'Failed to execute script "{ess_install_script}" to install ESS models')
    return TestIsaacROSPerceptorGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSPerceptorGraph(ROS2BenchmarkTest):
    """Live performance test for Isaac ROS Perceptor Graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Perceptor Graph Live Benchmark',
        benchmark_mode=BenchmarkMode.LIVE,
        benchmark_duration=20,
        test_iterations=5,
        post_trial_run_wait_time_sec=10,
        collect_start_timestamps_from_monitors=True,
        monitor_info_list=[]
    )

    def test_benchmark(self):
        self.config.monitor_info_list = NITROS_MONITOR_UTILITY.get_monitor_info_list()
        self.run_benchmark()
