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
"""Performance test for Isaac ROS cuMotion node."""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_moveit_benchmark import DifficultyMode, RobotGroup
from isaac_ros_moveit_benchmark import PlanningPipeline, TestPlanner

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS cuMotion node."""
    TestPlanner.config.benchmark_name = 'Isaac ROS cuMotion Node (Hard) Benchmark'
    TestPlanner.planning_pipline = PlanningPipeline.ISAAC_ROS_CUMOTION.value
    TestPlanner.diffiulty_mode = DifficultyMode.HARD.value
    TestPlanner.robot_group_name = RobotGroup.FRANKA.value

    cumotion_planner_node = Node(
            package='isaac_ros_cumotion',
            executable='cumotion_planner_node',
            name='cumotion_planner_node',
            output='screen',
            arguments=['--ros-args', '--log-level', 'error'],
            parameters=[
                {'robot': 'franka.yml'},
                {'time_dilation_factor': 1.0},
                {'tool_frame': 'panda_link8'},
            ]
    )

    isaac_ros_cumotion_examples_dir = os.path.join(
        get_package_share_directory('isaac_ros_cumotion_examples'), 'launch')
    cumotion_move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([isaac_ros_cumotion_examples_dir, '/franka.launch.py'])
    )

    return [
        cumotion_planner_node,
        cumotion_move_group_launch
    ]


def generate_test_description():
    return TestPlanner.generate_test_description_with_nsys(launch_setup)
