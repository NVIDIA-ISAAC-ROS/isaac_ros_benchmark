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

import isaac_ros_ess_benchmark.ess_model_utility as ess_model_utility

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode

# Input image resolution
HAWK_RESOLUTION = {'width': 1920, 'height': 1200}


def launch_setup(context, *args, **kwargs):
    print('Loading ESS depth graph configs...')

    container_name = LaunchConfiguration('container_name')
    node_namespace = LaunchConfiguration('node_namespace')
    type_negotiation_duration_s = LaunchConfiguration('type_negotiation_duration_s')

    ess_model_type = ''
    if LaunchConfigurationEquals('ess_model_type', 'full').evaluate(context):
        print('\tFull ESS model was selected')
        ess_model_type = 'full'
    elif LaunchConfigurationEquals('ess_model_type', 'light').evaluate(context):
        print('\tLight ESS model was selected')
        ess_model_type = 'light'
    else:
        raise ValueError('Unrecognized ess_model_type: {}'.format(
            LaunchConfiguration('ess_model_type').perform(context)))

    ess_resolution = ess_model_utility.get_mode_resolution(ess_model_type)

    engine_file_path = LaunchConfiguration('engine_file_path').perform(context)
    if LaunchConfigurationEquals('engine_file_path', '').evaluate(context):
        _, engine_file_path = ess_model_utility.get_model_paths(ess_model_type)
    print('\tUse ESS engine file path: {}'.format(engine_file_path))

    ess_throttler_skip = LaunchConfiguration('ess_throttler_skip')
    print('\tSet ESS throttler_skip to {}'.format(ess_throttler_skip.perform(context)))

    left_rectify_node = ComposableNode(
        name='LeftRectifyNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': HAWK_RESOLUTION['width'],
            'output_height': HAWK_RESOLUTION['height'],
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info_rect')
        ]
    )

    right_rectify_node = ComposableNode(
        name='RightRectifyNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': HAWK_RESOLUTION['width'],
            'output_height': HAWK_RESOLUTION['height'],
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info_rect')
        ]
    )

    left_resize_node = ComposableNode(
        name='LeftResizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': ess_resolution['width'],
            'output_height': ess_resolution['height'],
            'keep_aspect_ratio': True,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('camera_info', 'left/camera_info_rect'),
            ('image', 'left/image_rect'),
            ('resize/camera_info', 'left/camera_info_resize'),
            ('resize/image', 'left/image_resize')
        ]
    )

    right_resize_node = ComposableNode(
        name='RightResizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': ess_resolution['width'],
            'output_height': ess_resolution['height'],
            'keep_aspect_ratio': True,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('camera_info', 'right/camera_info_rect'),
            ('image', 'right/image_rect'),
            ('resize/camera_info', 'right/camera_info_resize'),
            ('resize/image', 'right/image_resize')
        ]
    )

    disparity_node = ComposableNode(
        name='ESSDisparityNode',
        namespace=node_namespace,
        package='isaac_ros_ess',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_layer_width': ess_resolution['width'],
            'input_layer_height': ess_resolution['height'],
            'throttler_skip': ess_throttler_skip,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('left/camera_info', 'left/camera_info_resize'),
            ('left/image_rect', 'left/image_resize'),
            ('right/camera_info', 'right/camera_info_resize'),
            ('right/image_rect', 'right/image_resize')
        ]
    )

    disparity_to_depth_node = ComposableNode(
        name='DisparityToDepthNode',
        namespace=node_namespace,
        package='isaac_ros_stereo_image_proc',
        plugin='nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode',
        parameters=[{
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
    )

    load_nodes = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=[
            left_rectify_node,
            right_rectify_node,
            left_resize_node,
            right_resize_node,
            disparity_node,
            disparity_to_depth_node,
        ],
    )

    return [load_nodes]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'container_name',
            description='Container name',
            default_value='container',
        ),
        DeclareLaunchArgument(
            'node_namespace',
            description='Node namespace',
            default_value='defaul_node_namespace',
        ),
        DeclareLaunchArgument(
            'type_negotiation_duration_s',
            description='Duration of the NITROS type negotiation.',
            default_value='5',
        ),
        DeclareLaunchArgument(
            'ess_model_type',
            description='Select ESS model type from "full", "light"',
            default_value='full',
        ),
        DeclareLaunchArgument(
            'engine_file_path',
            description='The absolute path to the ESS engine plan.',
            default_value='',
        ),
        DeclareLaunchArgument(
            'ess_throttler_skip',
            description='Frame skip setting for the ESS node.',
            default_value='0',
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
