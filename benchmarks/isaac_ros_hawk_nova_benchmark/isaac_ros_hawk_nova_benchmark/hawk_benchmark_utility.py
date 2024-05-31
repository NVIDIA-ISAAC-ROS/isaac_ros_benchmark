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

def create_hawk_monitors(
        nitros_monitor_utility, test_namespace, node_namespace=None, monitor_topics=None):
    monitor_nodes = []
    title_prefix = f'({node_namespace}) ' if node_namespace else ''
    node_name_prefix = f'{node_namespace}_' if node_namespace else ''
    # left image
    if monitor_topics is None or 'left/image_raw' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'left/image_raw',
            f'{title_prefix}Left Hawk Image',
            monitor_name=f'{node_name_prefix}LeftHawkImageMonitorNode',
            message_key_match=True
        ))
    # right image
    if monitor_topics is None or 'right/image_raw' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'right/image_raw',
            f'{title_prefix}Right Hawk Image',
            monitor_name=f'{node_name_prefix}RightHawkImageMonitorNode',
            message_key_match=True
        ))
    # left camera_info
    if monitor_topics is None or 'left/camera_info' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'left/camera_info',
            f'{title_prefix}Left Hawk Camera Info',
            monitor_name=f'{node_name_prefix}LeftHawkCameraInfoMonitorNode',
            message_key_match=True
        ))
    # right camera_info
    if monitor_topics is None or 'right/camera_info' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'right/camera_info',
            f'{title_prefix}Right Hawk Camera Info',
            monitor_name=f'{node_name_prefix}RightHawkCameraInfoMonitorNode',
            message_key_match=True
        ))
    return monitor_nodes
