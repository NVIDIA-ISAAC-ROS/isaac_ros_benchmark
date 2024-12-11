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

from enum import Enum

import numbers

from curobo.types.state import JointState as cuJointState

from moveit_msgs.msg import MoveItErrorCodes

import numpy

import rclpy

from rclpy.duration import Duration

import torch


class PlannerPerformanceMetrics(Enum):
    """Planner performance metrics."""

    MAX_PLANNING_TIME = 'Max. Planning Time (ms)'
    MIN_PLANNING_TIME = 'Min. Planning Time (ms)'
    MEAN_PLANNING_TIME = 'Mean Planning Time (ms)'
    MEDIAN_PLANNING_TIME = 'Median Planning Time (ms)'

    MAX_MOTION_TIME = 'Max. Motion Time (ms)'
    MIN_MOTION_TIME = 'Min. Motion Time (ms)'
    MEAN_MOTION_TIME = 'Mean Motion Time (ms)'
    MEDIAN_MOTION_TIME = 'Median Motion Time (ms)'

    MAX_JERK = 'Max. Jerk'
    MIN_JERK = 'Min. Jerk'
    MEAN_JERK = 'Mean Jerk'
    MEDIAN_JERK = 'Median Jerk'

    MAX_PATH_LENGTH = 'Max. Path Length (m)'
    MIN_PATH_LENGTH = 'Min. Path Length (m)'
    MEAN_PATH_LENGTH = 'Mean Path Length (m)'
    MEDIAN_PATH_LENGTH = 'Median Path Length (m)'

    PLANNING_SUCCESS_RATE = 'Planning Success Rate (%)'


class PlannerPerformanceCalculator():
    """Calculator that planner performance with planner metrics."""

    def __init__(self, config: dict = {}) -> None:
        """Initialize the calculator."""
        self.config = config
        self._report_prefix = config.get('report_prefix', '')
        self._logger = None
        self._perf_data_list = []

    def set_logger(self, logger):
        """Set logger that enables to print log messages."""
        self._logger = logger

    def get_logger(self):
        """Get logger for printing log messages."""
        if self._logger is not None:
            return self._logger
        return rclpy.logging.get_logger(self.__class__.__name__)

    def get_info(self):
        """Return a dict containing information for loading this calculator class."""
        info = {}
        info['module_name'] = self.__class__.__module__
        info['class_name'] = self.__class__.__name__
        info['config'] = self.config
        return info

    def reset(self):
        """Reset the calculator state."""
        self._perf_data_list.clear()

    def calculate_performance(self, planner_response_list) -> dict:
        """Calculate performance based on planner action service response."""
        perf_data = {}

        motion_time = []

        planning_time = []
        path_length = []
        jerk = []
        success_count = 0
        for planner_response in planner_response_list:
            if planner_response.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            success_count += 1

            q_traj = [p.positions for p in
                      planner_response.planned_trajectory.joint_trajectory.points]
            q_traj_vel = [p.velocities for p in
                          planner_response.planned_trajectory.joint_trajectory.points]
            q_traj_acc = [p.accelerations for p in
                          planner_response.planned_trajectory.joint_trajectory.points]

            js = cuJointState(position=torch.as_tensor(q_traj, device='cuda'),
                              velocity=torch.as_tensor(q_traj_vel, device='cuda'),
                              acceleration=torch.as_tensor(q_traj_acc, device='cuda'))
            path_length_t = torch.sum(
                torch.linalg.norm(
                    (torch.roll(js.position, -1, dims=-2) - js.position)[..., :-1, :], dim=-1,
                )
            )
            path_length.append(path_length_t.item())

            dt_values = []
            jerk_values = []
            max_jerk_limit = 500  # Set in joint_limits.yml
            times = numpy.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in
                                 planner_response.planned_trajectory.joint_trajectory.points])
            dt_values = numpy.diff(times)
            dt_values = torch.tensor(dt_values, device='cuda')
            q_traj_acc = numpy.array([p.accelerations for p in
                                      planner_response.planned_trajectory.joint_trajectory.points])
            q_traj_acc = torch.tensor(q_traj_acc, device='cuda')

            jerk_values = (q_traj_acc[1:] - q_traj_acc[:-1]) / dt_values.unsqueeze(1)
            jerk_padded = torch.cat((torch.zeros(
                1, len(planner_response.planned_trajectory.joint_trajectory.joint_names),
                device='cuda'), jerk_values), dim=0)
            js.jerk = jerk_padded

            max_jerk = torch.max(torch.abs(js.jerk)).item()

            # Clamp until we find a solution to the last timestep dt mismatch across a trajectory
            if max_jerk > max_jerk_limit:
                max_jerk = max_jerk_limit
            jerk.append(max_jerk)

            duration = Duration.from_msg(
                planner_response.planned_trajectory.joint_trajectory.points[-1].time_from_start)

            motion_time.append(duration.nanoseconds * 1e-9)
            planning_time.append(planner_response.planning_time * 1000)

        # Planning success rate
        planning_success_rate = 100 * success_count / len(planner_response_list)
        perf_data[PlannerPerformanceMetrics.PLANNING_SUCCESS_RATE] = planning_success_rate

        if planning_success_rate > 0:
            # Calculate planning time metrics
            perf_data[PlannerPerformanceMetrics.MAX_PLANNING_TIME] = max(planning_time)
            perf_data[PlannerPerformanceMetrics.MIN_PLANNING_TIME] = min(planning_time)
            perf_data[PlannerPerformanceMetrics.MEAN_PLANNING_TIME] = numpy.mean(planning_time)
            perf_data[PlannerPerformanceMetrics.MEDIAN_PLANNING_TIME] = numpy.median(planning_time)

            # Calculate motion time metrics
            perf_data[PlannerPerformanceMetrics.MAX_MOTION_TIME] = max(motion_time)
            perf_data[PlannerPerformanceMetrics.MIN_MOTION_TIME] = min(motion_time)
            perf_data[PlannerPerformanceMetrics.MEAN_MOTION_TIME] = numpy.mean(motion_time)
            perf_data[PlannerPerformanceMetrics.MEDIAN_MOTION_TIME] = numpy.median(motion_time)

            # Calculate path length metrics
            perf_data[PlannerPerformanceMetrics.MAX_PATH_LENGTH] = max(path_length)
            perf_data[PlannerPerformanceMetrics.MIN_PATH_LENGTH] = min(path_length)
            perf_data[PlannerPerformanceMetrics.MEAN_PATH_LENGTH] = numpy.mean(path_length)
            perf_data[PlannerPerformanceMetrics.MEDIAN_PATH_LENGTH] = numpy.median(path_length)

            # Calculate jerk metrics
            perf_data[PlannerPerformanceMetrics.MAX_JERK] = max(jerk)
            perf_data[PlannerPerformanceMetrics.MIN_JERK] = min(jerk)
            perf_data[PlannerPerformanceMetrics.MEAN_JERK] = numpy.mean(jerk)
            perf_data[PlannerPerformanceMetrics.MEDIAN_JERK] = numpy.median(jerk)

        # Store the current perf results to be concluded later
        self._perf_data_list.append(perf_data)

        if self._report_prefix != '':
            return {self._report_prefix: perf_data}
        return perf_data

    def conclude_performance(self) -> dict:
        """Calculate final statistical performance outcome based on all results."""
        if len(self._perf_data_list) == 0:
            self.get_logger().warn('No prior performance measurements to conclude')
            return {}

        MEAN_METRICS = [
            PlannerPerformanceMetrics.PLANNING_SUCCESS_RATE,
            # Mean metrics
            PlannerPerformanceMetrics.MEAN_PLANNING_TIME,
            PlannerPerformanceMetrics.MEAN_MOTION_TIME,
            PlannerPerformanceMetrics.MEAN_JERK,
            PlannerPerformanceMetrics.MEAN_PATH_LENGTH,
            # Median metrics
            PlannerPerformanceMetrics.MEDIAN_PLANNING_TIME,
            PlannerPerformanceMetrics.MEDIAN_MOTION_TIME,
            PlannerPerformanceMetrics.MEDIAN_JERK,
            PlannerPerformanceMetrics.MEDIAN_PATH_LENGTH,
        ]
        MAX_METRICS = [
            PlannerPerformanceMetrics.MAX_PLANNING_TIME,
            PlannerPerformanceMetrics.MAX_MOTION_TIME,
            PlannerPerformanceMetrics.MAX_JERK,
            PlannerPerformanceMetrics.MAX_PATH_LENGTH,
        ]
        MIN_METRICS = [
            PlannerPerformanceMetrics.MIN_PLANNING_TIME,
            PlannerPerformanceMetrics.MIN_MOTION_TIME,
            PlannerPerformanceMetrics.MIN_JERK,
            PlannerPerformanceMetrics.MIN_PATH_LENGTH,
        ]

        final_perf_data = {}
        for metric in PlannerPerformanceMetrics:
            metric_value_list = [perf_data.get(metric, None) for perf_data in self._perf_data_list]
            if not all(isinstance(value, numbers.Number) for value in metric_value_list):
                continue

            if metric in MEAN_METRICS:
                final_perf_data[metric] = sum(metric_value_list)/len(metric_value_list)
            elif metric in MAX_METRICS:
                final_perf_data[metric] = max(metric_value_list)
            elif metric in MIN_METRICS:
                final_perf_data[metric] = min(metric_value_list)
            else:
                final_perf_data[metric] = 'INVALID VALUES: NO CONCLUDED METHOD ASSIGNED'

        self.reset()
        if self._report_prefix != '':
            return {self._report_prefix: final_perf_data}
        return final_perf_data
