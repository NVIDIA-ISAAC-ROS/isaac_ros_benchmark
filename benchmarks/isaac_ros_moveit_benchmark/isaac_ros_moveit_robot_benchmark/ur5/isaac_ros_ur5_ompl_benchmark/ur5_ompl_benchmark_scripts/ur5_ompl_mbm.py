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
import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_moveit_benchmark import PlanningPipeline, RobotGroup, TestPlannerMBM

from launch.actions import GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource


TESTING_SLEEP_TIME = 0.05


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking OMPL planners."""
    TestPlannerMBM.config.benchmark_name = 'OMPL Pipeline Planners MBM Benchmark'
    TestPlannerMBM.planning_pipline = PlanningPipeline.OMPL.value
    TestPlannerMBM.robot_group_name = RobotGroup.UR5_ROBOTIQ_85.value

    robot_moveit_launch_dir = os.path.join(
        get_package_share_directory('ur5_gripper_moveit_config'), 'launch')
    ompl_move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_moveit_launch_dir, '/demo.launch.py']),
        launch_arguments={
            'default_planning_pipeline': 'ompl',
            'use_rviz': 'False'}.items()
    )

    ompl_move_group_action = GroupAction(
        actions=[
            SetEnvironmentVariable(name='DISPLAY', value='""'),
            ompl_move_group_launch,
        ]
    )

    return [
        ompl_move_group_action,
    ]


def generate_test_description():
    return TestPlannerMBM.generate_test_description_with_nsys(launch_setup)
