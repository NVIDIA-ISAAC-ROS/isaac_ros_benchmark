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
"""Performance test base for planners."""

import abc

from enum import Enum

import time

# CuRobo
from curobo.geom.sdf.world import WorldConfig

from geometry_msgs.msg import Pose, Transform

from isaac_ros_moveit_benchmark.planner_performance_calculator import PlannerPerformanceCalculator

from moveit_msgs.action import MoveGroup

from moveit_msgs.msg import CollisionObject, Constraints, LinkPadding, MotionPlanRequest, \
    OrientationConstraint, PlanningOptions, \
    PlanningScene, PositionConstraint, RobotState

import rclpy

from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


class DifficultyMode(Enum):
    """Difficulty mode."""

    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'

    @classmethod
    def list_all(cls):
        return [e.value for e in cls]


class PlanningPipeline(Enum):
    """Planning pipeline."""

    OMPL = 'ompl'
    ISAAC_ROS_CUMOTION = 'isaac_ros_cumotion'

    @classmethod
    def list_all(cls):
        return [e.value for e in cls]


class RobotGroup(Enum):
    """Robot group."""

    FRANKA = 'panda_arm'
    UR = 'ur_manipulator'
    UR5_ROBOTIQ_85 = 'ur5_arm'

    @classmethod
    def list_all(cls):
        return [e.value for e in cls]


class RobotType(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_link_names(self):
        pass

    @abc.abstractmethod
    def get_ee_link_name(self):
        pass

    @abc.abstractmethod
    def get_joint_names(self):
        pass

    @property
    @abc.abstractmethod
    def home_state(self):
        pass

    @home_state.setter
    @abc.abstractmethod
    def home_state(self, value):
        pass

    @property
    @abc.abstractmethod
    def goal_pose_list(self):
        pass

    @goal_pose_list.setter
    @abc.abstractmethod
    def goal_pose_list(self):
        pass


class Franka(RobotType):

    def __init__(self):
        self._home_state = []
        self._goal_pose_list = []

    def get_link_names(self):
        return ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
                'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8',
                'panda_leftfinger', 'panda_rightfinger']

    def get_ee_link_name(self):
        return 'panda_link8'

    def get_joint_names(self):
        return ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5',
                'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

    @property
    def home_state(self):
        return self._home_state

    @home_state.setter
    def home_state(self, value):
        self._home_state = value

    @property
    def goal_pose_list(self):
        return self._goal_pose_list

    @goal_pose_list.setter
    def goal_pose_list(self, value):
        self._goal_pose_list = value


class UR(RobotType):

    def get_link_names(self):
        return ['base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link',
                'wrist_2_link', 'wrist_3_link']

    def get_ee_link_name(self):
        return 'wrist_3_link'

    def get_joint_names(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                'wrist_2_joint', 'wrist_3_joint']

    @property
    def home_state(self):
        return self._home_state

    @home_state.setter
    def home_state(self, value):
        self._home_state = value

    @property
    def goal_pose_list(self):
        return self._goal_pose

    @goal_pose_list.setter
    def goal_pose_list(self, value):
        self._goal_pose = value


class UR5_Robotiq_85(RobotType):

    def get_link_names(self):
        return ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link',
                'wrist_2_link', 'wrist_3_link', 'fts_robotside',
                'robotiq_85_base_link',
                'robotiq_85_left_knuckle_link',
                'robotiq_85_left_finger_link',
                'robotiq_85_left_inner_knuckle_link',
                'robotiq_85_left_finger_tip_link',
                'robotiq_85_right_inner_knuckle_link',
                'robotiq_85_right_finger_tip_link',
                'robotiq_85_right_knuckle_link',
                'robotiq_85_right_finger_link']

    def get_ee_link_name(self):
        return 'wrist_3_link'

    def get_joint_names(self):
        return ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                'wrist_2_joint', 'wrist_3_joint']

    @property
    def home_state(self):
        return self._home_state

    @home_state.setter
    def home_state(self, value):
        self._home_state = value

    @property
    def goal_pose_list(self):
        return self._goal_pose

    @goal_pose_list.setter
    def goal_pose_list(self, value):
        self._goal_pose = value


class TestPlanner(ROS2BenchmarkTest):
    """Performance test for planners."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Planner Benchmark',
        test_iterations=5,
        pre_trial_run_wait_time_sec=60.0,
    )

    planning_pipline = PlanningPipeline.ISAAC_ROS_CUMOTION.value
    diffiulty_mode = DifficultyMode.EASY.value
    robot_group_name = RobotGroup.FRANKA.value

    def test_benchmark(self):
        self.result = None

        cb_group = MutuallyExclusiveCallbackGroup()

        self._action_client = ActionClient(self.node, MoveGroup, '/move_action',
                                           callback_group=cb_group)

        self.validate_benchmark_setup()

        self.robot = self.init_robot(self.robot_group_name)

        self._scene_publisher = self.node.create_publisher(
            PlanningScene, '/monitored_planning_scene', 5)

        self.setup_planning_problem()
        planner_id_list = self.get_available_planners(self.planning_pipline)

        final_perf_results = {}
        for planner in planner_id_list:
            self.get_logger().info(f'Evaluating planning_pipeline: {self.planning_pipline} '
                                   f'with planner: {planner}')
            # Reset resource profiler
            self._resource_profiler.reset()

            planner_perf_calculator = PlannerPerformanceCalculator()
            for i in range(self.config.test_iterations):
                performance_results = {}

                # Start resource profiler
                self._resource_profiler.stop_profiling()
                self._resource_profiler.start_profiling(
                    self.config.resource_profiling_interval_sec)
                self.get_logger().info('Resource profiling started.')

                planner_response_list = []
                for pose in self.robot.goal_pose_list:
                    planner_response_list.append(
                        self.send_ee_goal(
                            self._planning_options, self.robot.home_state, pose, planner))

                # Stop resource profiler
                self._resource_profiler.stop_profiling()
                self.get_logger().info('Resource profiling stopped.')

                # Calculate performance
                performance_results = {}

                # Planner performance
                performance_results.update(
                    planner_perf_calculator.calculate_performance(planner_response_list))

                # Resource profiler results
                performance_results.update(self._resource_profiler.get_results())

                self.print_report(performance_results, sub_heading=f'{planner} - #{i+1}')

            # Conclude perf results for a planner after all iterations
            final_planner_perf_results = {}
            final_planner_perf_results.update(planner_perf_calculator.conclude_performance())
            final_planner_perf_results.update(self._resource_profiler.conclude_results())
            self.print_report(final_planner_perf_results, sub_heading=f'{planner} - Summary')

            final_perf_results[f'{planner}'] = final_planner_perf_results

        final_report = self.construct_final_report(final_perf_results)
        self.print_report(final_report, sub_heading='Final Report')
        print('\r\n')
        self.print_report(final_report, sub_heading='Final Report', print_func=print)
        print('\r\n')
        self.export_report(final_report)

    def validate_benchmark_setup(self):
        if self.diffiulty_mode not in DifficultyMode.list_all():
            self.get_logger().error(f'Invalid diffiulty mode. Please select from \
                                    {DifficultyMode.list_all()}')
            raise ValueError(f'Invalid diffiulty mode. Please select from \
                             {DifficultyMode.list_all()}')

        if self.planning_pipline not in PlanningPipeline.list_all():
            self.get_logger().error(f'Invalid planning pipeline. Please select from \
                                    {PlanningPipeline.list_all()}')
            raise ValueError(f'Invalid planning pipeline. Please select from \
                                    {PlanningPipeline.list_all()}')

        if self.robot_group_name not in RobotGroup.list_all():
            self.get_logger().error(f'Invalid robot group name. Please select from \
                                    {RobotGroup.list_all()}')
            raise ValueError(f'Invalid robot group name. Please select from \
                                    {RobotGroup.list_all()}')

    def init_robot(self, robot_group_name):
        if robot_group_name == RobotGroup.FRANKA.value:
            return Franka()
        elif robot_group_name == RobotGroup.UR.value:
            return UR()
        elif robot_group_name == RobotGroup.UR5_ROBOTIQ_85.value:
            return UR5_Robotiq_85()
        else:
            raise ValueError(f'Invalid robot group name. Please select from \
                {RobotGroup.list_all()}')

    def get_available_planners(self, planning_pipline):
        # Get available planners
        available_planners = []
        if self.planning_pipline == 'ompl':
            available_planners = [
                'RRTConnectkConfigDefault',
                'RRTkConfigDefault',
                'ESTkConfigDefault',
                'LazyPRMstarkConfigDefault',
                'SPARSkConfigDefault',
                'SPARStwokConfigDefault',
                'TRRTkConfigDefault',
                'PRMkConfigDefault',
                'PRMstarkConfigDefault',
                'FMTkConfigDefault',
                'BFMTkConfigDefault',
                'BiTRRTkConfigDefault',
                'LBTRRTkConfigDefault',
                'BiESTkConfigDefault',
                'RRTstarkConfigDefault',
            ]
        elif planning_pipline == 'isaac_ros_cumotion':
            available_planners = ['cuMotion']

        self.get_logger().debug(
            f'Available planners for {self.planning_pipline}: {str(available_planners)}')
        return available_planners

    def setup_planning_problem(self):
        # Get diffiulty mode
        if self.diffiulty_mode == DifficultyMode.EASY.value:
            x_d = 0.7
        elif self.diffiulty_mode == DifficultyMode.MEDIUM.value:
            x_d = 0.6
        else:
            x_d = 0.5

        # Set world
        world = {
            'cuboid': {
                'bar1': {
                    'dims': [0.5, 0.05, 1.0],
                    'pose': [x_d, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                },
                'bar2': {
                    'dims': [0.5, 1.0, 0.05],
                    'pose': [x_d, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                },
                'bar3': {
                    'dims': [0.5, 0.05, 1.0],
                    'pose': [x_d, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
                },
                'bar4': {
                    'dims': [0.5, 0.05, 1.0],
                    'pose': [x_d, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0],
                },
                'bar5': {
                    'dims': [0.5, 1.1, 0.05],
                    'pose': [x_d, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                },
                'floor': {
                    'dims': [2.0, 2.0, 0.1],
                    'pose': [0.0, -0.0, -0.06, 1.0, 0.0, 0.0, 0.0],
                },
            }
        }

        # Convert world to planning options
        collision_objects = WorldConfig.from_dict(world).cuboid
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

        self._scene_publisher.publish(self._planning_options.planning_scene_diff)

        # This is the home state for the franka robot
        self.robot.home_state = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04]

        # Set end effector goal poses in cartesian space
        self.robot.goal_pose_list = [
            [0.55, -0.25, 0.25, 0.0, 0.707, 0.0, 0.707],
            [0.55, -0.25, 0.75, 0.0, 0.707, 0.0, 0.707],
            [0.55, 0.25, 0.75, 0.0, 0.707, 0.0, 0.707],
            [0.55, 0.25, 0.25, 0.0, 0.707, 0.0, 0.707]
        ]

    def create_request(self, planning_option, planner_id, group_name, ee_link_name,
                       joint_start_positions, ee_pose):
        goal_msg = MoveGroup.Goal()
        motion_plan_request = MotionPlanRequest()
        motion_plan_request.allowed_planning_time = 30.0
        motion_plan_request.max_velocity_scaling_factor = 1.0
        motion_plan_request.max_acceleration_scaling_factor = 1.0
        motion_plan_request.planner_id = planner_id
        motion_plan_request.group_name = group_name
        start_state = RobotState()
        joint_state = JointState()
        joint_state.name = self.robot.get_joint_names()
        joint_state.position = joint_start_positions
        start_state.joint_state = joint_state
        start_state.is_diff = False
        start_state.multi_dof_joint_state.joint_names = ['virtual_joint']
        start_state.multi_dof_joint_state.header.frame_id = 'world'
        transform = Transform()
        transform.translation.x = 0.0
        transform.translation.y = 0.0
        transform.translation.z = 0.0
        transform.rotation.w = 1.0
        transform.rotation.x = 0.0
        transform.rotation.y = 0.0
        transform.rotation.z = 0.0
        start_state.multi_dof_joint_state.transforms = [transform]
        motion_plan_request.start_state = start_state

        constraints = Constraints()
        # Add position constraints
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = 'world'
        position_constraint.link_name = ee_link_name
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0
        position_constraint.weight = 1.0
        goal_bbox = SolidPrimitive()
        goal_bbox.type = SolidPrimitive.BOX
        goal_bbox.dimensions = [0.01, 0.01, 0.01]
        position_constraint.constraint_region.primitives.append(goal_bbox)
        goal_pose = Pose()
        goal_pose.position.x = float(ee_pose[0])
        goal_pose.position.y = float(ee_pose[1])
        goal_pose.position.z = float(ee_pose[2])
        goal_pose.orientation.w = float(ee_pose[3])
        goal_pose.orientation.x = float(ee_pose[4])
        goal_pose.orientation.y = float(ee_pose[5])
        goal_pose.orientation.z = float(ee_pose[6])
        position_constraint.constraint_region.primitive_poses.append(goal_pose)
        constraints.position_constraints.extend([position_constraint])

        # Add orientation constraints
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = 'world'
        orientation_constraint.link_name = ee_link_name
        orientation_constraint.weight = 1.0
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.orientation.w = ee_pose[3]
        orientation_constraint.orientation.x = ee_pose[4]
        orientation_constraint.orientation.y = ee_pose[5]
        orientation_constraint.orientation.z = ee_pose[6]
        constraints.orientation_constraints.extend([orientation_constraint])

        motion_plan_request.goal_constraints.append(constraints)

        goal_msg.request = motion_plan_request

        goal_msg.planning_options = planning_option
        goal_msg.planning_options.planning_scene_diff.robot_state = start_state

        for name in self.robot.get_link_names():
            link_padding = LinkPadding()
            link_padding.link_name = name
            link_padding.padding = 0.0
            goal_msg.planning_options.planning_scene_diff.link_padding.append(link_padding)

        goal_msg.planning_options.plan_only = True

        return goal_msg

    def send_ee_goal(self, planning_option, joint_start_positions, ee_pose, planner_id):
        self.result = None

        goal_msg = self.create_request(planning_option, planner_id, self.robot_group_name,
                                       self.robot.get_ee_link_name(), joint_start_positions,
                                       ee_pose)

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        while not self.result:
            rclpy.spin_once(self.node)
            time.sleep(0.001)

        return self.result

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.result = future.result().result
        return
