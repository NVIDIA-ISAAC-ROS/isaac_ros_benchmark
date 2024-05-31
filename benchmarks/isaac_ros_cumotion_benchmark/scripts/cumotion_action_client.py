#!/usr/bin/env python3

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

from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.msg import MotionPlanRequest, PlanningOptions, RobotState
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from sensor_msgs.msg import JointState


class MoveGroupActionClient(Node):

    def __init__(self):
        super().__init__('move_group_action_client')
        self._action_client = ActionClient(self, MoveGroup, 'cumotion/move_group')
        # Publisher setup, using `PoseStamped` since its the most simple stamped type
        self.success_publisher = self.create_publisher(
            PoseStamped, 'cumotion_planning_success', 10)
        self.joint_names = self.declare_parameter(
            'joint_names').get_parameter_value().string_array_value
        self.joint_start_positions = self.declare_parameter(
            'joint_start_positions').get_parameter_value().double_array_value
        self.joint_end_positions = self.declare_parameter(
            'joint_end_positions').get_parameter_value().double_array_value
        self.sleep_time = self.declare_parameter(
            'sleep_time', 0.1).get_parameter_value().double_value
        self.send_goal(self.joint_names, self.joint_start_positions,
                       self.joint_end_positions, [0.1] * len(self.joint_names),
                       [0.1] * len(self.joint_names))

    def send_goal(
        self, joint_names, joint_start_positions,
            joint_end_positions, tolerances_above, tolerances_below):
        joint_constraints = []
        for joint_name, joint_position, tolerance_above, tolerance_below in zip(
                joint_names, joint_end_positions, tolerances_above, tolerances_below):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint_name
            joint_constraint.position = joint_position
            joint_constraint.tolerance_above = tolerance_above
            joint_constraint.tolerance_below = tolerance_below
            joint_constraint.weight = 1.0
            joint_constraints.append(joint_constraint)

        goal_msg = MoveGroup.Goal()
        motion_plan_request = MotionPlanRequest()
        start_state = RobotState()
        joint_state = JointState()
        joint_state.name = joint_names
        joint_state.position = joint_start_positions
        start_state.joint_state = joint_state
        motion_plan_request.start_state = start_state

        constraints = Constraints()
        constraints.joint_constraints.extend(joint_constraints)
        motion_plan_request.goal_constraints.append(constraints)

        goal_msg.request = motion_plan_request
        goal_msg.planning_options = PlanningOptions()

        self._action_client.wait_for_server()
        self.get_logger().info('Action server is ready.')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted, waiting for result :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == MoveItErrorCodes.SUCCESS:
            self.get_logger().info('Motion planning succeeded.')
            # Publish a success message
            success_msg = PoseStamped()
            success_msg.header.stamp = self.get_clock().now().to_msg()
            self.success_publisher.publish(success_msg)
            self.send_goal(self.joint_names, self.joint_start_positions,
                           self.joint_end_positions, [0.1] * len(self.joint_names),
                           [0.1] * len(self.joint_names))
            time.sleep(self.sleep_time)
        else:
            self.get_logger().info('Motion planning failed.')


def main(args=None):
    rclpy.init(args=args)
    action_client = MoveGroupActionClient()

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Interrupted by user, shutting down.')
    finally:
        action_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
