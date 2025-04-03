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
import time
from typing import Any, Dict, List

from curobo.geom.sdf.world import WorldConfig

from isaac_ros_moveit_benchmark.mbm_loader import MBMLoader
from isaac_ros_moveit_benchmark.planner_benchmark_base import TestPlanner
from isaac_ros_moveit_benchmark.planner_performance_calculator import PlannerPerformanceCalculator

from moveit_msgs.action import MoveGroup

from moveit_msgs.msg import CollisionObject, PlanningOptions, PlanningScene

from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from shape_msgs.msg import SolidPrimitive

from tqdm import tqdm


class TestPlannerMBM(TestPlanner):
    """Performance test for planners on MBM dataset."""

    def test_benchmark(self) -> None:
        self.config.test_iterations = 1

        self.result = None

        cb_group = MutuallyExclusiveCallbackGroup()

        self._action_client = ActionClient(self.node, MoveGroup, '/move_action',
                                           callback_group=cb_group)

        self.validate_benchmark_setup()

        self.robot = self.init_robot(self.robot_group_name)

        self._scene_publisher = self.node.create_publisher(
            PlanningScene, '/monitored_planning_scene', 5)

        loader = MBMLoader(self.robot_group_name)
        file_paths = [loader.motion_benchmaker_raw][:]

        planner_id_list = self.get_available_planners(self.planning_pipline)
        final_perf_results = {}

        for planner in planner_id_list:
            self.get_logger().info(f'Evaluating planning_pipeline: {self.planning_pipline} '
                                   f'with planner: {planner}')
            # Reset resource profiler
            self._resource_profiler.reset()
            planner_perf_calculator = PlannerPerformanceCalculator()

            for i in range(self.config.test_iterations):
                # Calculate performance
                performance_results = {}

                self._resource_profiler.stop_profiling()
                self._resource_profiler.start_profiling(
                    self.config.resource_profiling_interval_sec)
                self.get_logger().info('Resource profiling started.')
                planner_response_list = []

                for file_path in file_paths:
                    problems = file_path()
                    for _, scene_problems in tqdm(problems.items()):
                        for _, scene in enumerate(tqdm(scene_problems, leave=False)):
                            self.setup_mbm_planning_problem(scene)
                            self.get_logger().info(
                                f'Evaluating planning_pipeline: {self.planning_pipline}'
                                'with planner: {planner}')
                            pose = self.robot.goal_pose_list[0]
                            planner_response = self.send_ee_goal(self._planning_options,
                                                                 self.robot.home_state,
                                                                 pose,
                                                                 planner)
                            planner_response_list.append(planner_response)
                            time.sleep(5)
                self._resource_profiler.stop_profiling()
                self.get_logger().info('Resource profiling stopped.')

                # Planner performance
                performance_results.update(
                    planner_perf_calculator.calculate_performance(planner_response_list))

                # Resource profiler results
                performance_results.update(self._resource_profiler.get_results())

                self.print_report(performance_results, sub_heading=f'{planner} - #{i+1}')

            final_planner_perf_results = {}
            final_planner_perf_results.update(planner_perf_calculator.conclude_performance())
            self.print_report(final_planner_perf_results, sub_heading=f'{planner} - Summary')

            final_perf_results[f'{planner}'] = final_planner_perf_results

        final_report = self.construct_final_report(final_perf_results)
        self.print_report(final_report, sub_heading='Final Report')
        print('\r\n')
        self.print_report(final_report, sub_heading='Final Report', print_func=print)
        print('\r\n')
        self.export_report(final_report)

    def setup_mbm_planning_problem(self, problem: Dict[str, Any]) -> None:
        # Add the cuboidal base below the robot
        cube_count = len(problem['obstacles']['cuboid'])
        problem['obstacles']['cuboid'][f'cube{cube_count}'] = {
            'dims': [0.1, 0.1, 0.6],  # x, y, z
            'pose': [0.0, 0.0, -0.45, 1, 0, 0, 0.0]
        }

        collision_objects = WorldConfig.from_dict(
            problem['obstacles']).get_obb_world()
        self._planning_options = PlanningOptions()
        self._planning_options.plan_only = True
        self._planning_options.planning_scene_diff.is_diff = False
        for collision_object in collision_objects:
            collision_obj = CollisionObject()
            collision_obj.header.frame_id = 'world'
            collision_obj.operation = CollisionObject.ADD
            collision_obj.id = collision_object.name
            collision_obj.pose.position.x = float(collision_object.pose[0])
            collision_obj.pose.position.y = float(collision_object.pose[1])
            collision_obj.pose.position.z = float(collision_object.pose[2])
            collision_obj.pose.orientation.w = float(collision_object.pose[3])
            collision_obj.pose.orientation.x = float(collision_object.pose[4])
            collision_obj.pose.orientation.y = float(collision_object.pose[5])
            collision_obj.pose.orientation.z = float(collision_object.pose[6])
            solid_primitive = SolidPrimitive()
            solid_primitive.type = SolidPrimitive.BOX
            solid_primitive.dimensions = collision_object.dims
            collision_obj.primitives.append(solid_primitive)
            self._planning_options.planning_scene_diff.world.collision_objects.append(
                collision_obj)

        position = problem['goal_pose']['position_xyz']
        quaternion_wxyz = problem['goal_pose']['quaternion_wxyz']
        pose = position + quaternion_wxyz
        self._scene_publisher.publish(self._planning_options.planning_scene_diff)
        self.robot.home_state = problem['start']
        self.robot.goal_pose_list = [pose]

    def get_available_planners(self, planning_pipline: str) -> List[str]:
        # Get available planners
        available_planners = []
        if self.planning_pipline == 'ompl':
            available_planners = [
                'RRTConnectkConfigDefault',
                'RRTkConfigDefault',
                'TRRTkConfigDefault',
                'PRMkConfigDefault',
                'ESTkConfigDefault',
            ]
        elif planning_pipline == 'isaac_ros_cumotion':
            available_planners = ['cuMotion']

        self.get_logger().debug(
            f'Available planners for {self.planning_pipline}: {str(available_planners)}')
        return available_planners
