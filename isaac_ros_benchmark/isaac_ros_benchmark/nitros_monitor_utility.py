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


from typing import List

from launch_ros.descriptions import ComposableNode
from ros2_benchmark import BasicPerformanceCalculator
from ros2_benchmark import MonitorPerformanceCalculatorsInfo


class NitrosMonitorUtility:
    """Nitros monitor utility class."""

    def __init__(self):
        self._monitor_counter = 0
        self._monitor_node_list = []
        self._monitor_info_list = []

    def add_basic_perf_nitros_monitor(
            self, monitor_namespace, node_namespace, monitor_data_format, topic_name,
            report_prefix, monitor_name=None, message_key_match=True,
            use_nitros_type_monitor_sub=True):
        monitor_name = report_prefix if not monitor_name else monitor_name
        if node_namespace is None or node_namespace == '':
            node_namespace = monitor_namespace
        if node_namespace[0] == '/':
            monitor_topic = f'{node_namespace}/{topic_name}'
        else:
            monitor_topic = f'/{node_namespace}/{topic_name}'

        monitor_node = ComposableNode(
            name=monitor_name,
            namespace=monitor_namespace,
            package='isaac_ros_benchmark',
            plugin='isaac_ros_benchmark::NitrosMonitorNode',
            parameters=[{
                'monitor_index': self._monitor_counter,
                'monitor_data_format': monitor_data_format,
                'use_nitros_type_monitor_sub': use_nitros_type_monitor_sub,
            }],
            remappings=[('output', monitor_topic)]
        )
        self._monitor_info_list.append(
            MonitorPerformanceCalculatorsInfo(
                f'monitor_node{self._monitor_counter}',
                [BasicPerformanceCalculator({
                    'report_prefix': report_prefix,
                    'message_key_match': message_key_match
                })]
            )
        )
        self._monitor_node_list.append(monitor_node)
        self._monitor_counter += 1
        return monitor_node

    def get_monitor_info_list(self) -> List[MonitorPerformanceCalculatorsInfo]:
        return self._monitor_info_list

    def get_monitor_node_list(self) -> List[ComposableNode]:
        return self._monitor_node_list
