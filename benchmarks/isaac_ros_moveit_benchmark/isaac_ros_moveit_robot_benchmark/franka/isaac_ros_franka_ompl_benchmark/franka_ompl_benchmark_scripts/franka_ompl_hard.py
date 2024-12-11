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
"""Performance test for OMPL planners (hard)."""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_moveit_benchmark import DifficultyMode, RobotGroup
from isaac_ros_moveit_benchmark import PlanningPipeline, TestPlanner

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking OMPL planners."""
    TestPlanner.config.benchmark_name = 'OMPL Pipeline Planners (Hard) Benchmark'
    TestPlanner.planning_pipline = PlanningPipeline.OMPL.value
    TestPlanner.diffiulty_mode = DifficultyMode.HARD.value
    TestPlanner.robot_group_name = RobotGroup.FRANKA.value

    moveit2_tutorials_dir = os.path.join(
        get_package_share_directory('moveit2_tutorials'), 'launch')
    ompl_move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([moveit2_tutorials_dir, '/demo.launch.py'])
    )

    return [ompl_move_group_launch]


def generate_test_description():
    return TestPlanner.generate_test_description_with_nsys(launch_setup)
