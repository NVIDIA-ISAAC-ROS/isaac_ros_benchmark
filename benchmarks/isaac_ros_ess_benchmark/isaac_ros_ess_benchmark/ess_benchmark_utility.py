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


def create_ess_depth_graph_monitors(
        nitros_monitor_utility, test_namespace, node_namespace=None,
        monitor_topics=None, message_key_match=True):
    monitor_nodes = []
    title_prefix = f'({node_namespace}) ' if node_namespace else ''
    node_name_prefix = f'{node_namespace}_' if node_namespace else ''
    # rectified left image
    if monitor_topics is None or 'left/image_rect' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'left/image_rect',
            f'{title_prefix}Left Rectified Image',
            monitor_name=f'{node_name_prefix}LeftRectifiedImageMonitorNode',
            message_key_match=message_key_match
        ))
    # rectified right image
    if monitor_topics is None or 'right/image_rect' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'right/image_rect',
            f'{title_prefix}Right Rectified Image',
            monitor_name=f'{node_name_prefix}RightRectifiedImageMonitorNode',
            message_key_match=message_key_match
        ))
    # rectified left camera info
    if monitor_topics is None or 'left/camera_info_rect' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'left/camera_info_rect',
            f'{title_prefix}Left Rectified Camera Info',
            monitor_name=f'{node_name_prefix}LeftRectifiedCameraInfoMonitorNode',
            message_key_match=message_key_match
        ))
    # rectified right camera info
    if monitor_topics is None or 'right/camera_info_rect' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'right/camera_info_rect',
            f'{title_prefix}Right Rectified Camera Info',
            monitor_name=f'{node_name_prefix}RightRectifiedCameraInfoMonitorNode',
            message_key_match=message_key_match
        ))
    # resized left image
    if monitor_topics is None or 'left/image_resize' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'left/image_resize',
            f'{title_prefix}Left Resized Image',
            monitor_name=f'{node_name_prefix}LeftResizedImageMonitorNode',
            message_key_match=message_key_match
        ))
    # resized right image
    if monitor_topics is None or 'right/image_resize' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_rgb8', 'right/image_resize',
            f'{title_prefix}Right Resized Image',
            monitor_name=f'{node_name_prefix}RightResizedImageMonitorNode',
            message_key_match=message_key_match
        ))
    # resized left camera info
    if monitor_topics is None or 'left/camera_info_resize' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'left/camera_info_resize',
            f'{title_prefix}Left Resized Camera Info',
            monitor_name=f'{node_name_prefix}LeftResizedCameraInfoMonitorNode',
            message_key_match=message_key_match
        ))
    # resized right camera info
    if monitor_topics is None or 'right/camera_info_resize' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_camera_info', 'right/camera_info_resize',
            f'{title_prefix}Right Resized Camera Info',
            monitor_name=f'{node_name_prefix}RightResizedCameraInfoMonitorNode',
            message_key_match=message_key_match
        ))
    # disparity
    if monitor_topics is None or 'disparity' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_disparity_image_32FC1', 'disparity',
            f'{title_prefix}Disparity',
            monitor_name=f'{node_name_prefix}DisparityMonitorNode',
            message_key_match=message_key_match
        ))
    # depth
    if monitor_topics is None or 'depth' in monitor_topics:
        monitor_nodes.append(nitros_monitor_utility.add_basic_perf_nitros_monitor(
            test_namespace,
            node_namespace, 'nitros_image_32FC1', 'depth',
            f'{title_prefix}Depth',
            monitor_name=f'{node_name_prefix}DepthMonitorNode',
            message_key_match=message_key_match
        ))
    return monitor_nodes
