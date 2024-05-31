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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    print('Loading Hawk node...')

    container_name = LaunchConfiguration('container_name')
    node_namespace = LaunchConfiguration('node_namespace')
    type_negotiation_duration_s = LaunchConfiguration('type_negotiation_duration_s')

    correlated_timestamp_driver_node = None
    if LaunchConfigurationEquals(
            'create_correlated_timestamp_driver_node', 'True').evaluate(context):
        correlated_timestamp_driver_node = ComposableNode(
            name='CorrelatedTimestampDriver',
            namespace=node_namespace,
            package='isaac_ros_correlated_timestamp_driver',
            plugin='nvidia::isaac_ros::correlated_timestamp_driver::CorrelatedTimestampDriverNode',
            parameters=[{
                'use_time_since_epoch': False,
                'nvpps_dev_name': '/dev/nvpps0',
                'type_negotiation_duration_s': type_negotiation_duration_s,
            }],
            remappings=[
                ('correlated_timestamp', '/correlated_timestamp'),
            ]
        )
        print('\tAdded CorrelatedTimestampDriverNode')

    hawk_module_id = 5
    if LaunchConfigurationEquals('hawk_placement', 'front').evaluate(context):
        hawk_module_id = 5
    elif LaunchConfigurationEquals('hawk_placement', 'left').evaluate(context):
        hawk_module_id = 7
    elif LaunchConfigurationEquals('hawk_placement', 'right').evaluate(context):
        hawk_module_id = 2
    elif LaunchConfigurationEquals('hawk_placement', 'back').evaluate(context):
        hawk_module_id = 6
    else:
        raise ValueError('Unrecognized Hawk camera placement: {}'.format(
            LaunchConfiguration('hawk_placement').perform(context)))
    print('\tHawk camera (module_id={}) was selected'.format(hawk_module_id))

    hawk_node = ComposableNode(
        name='HawkNode',
        namespace=node_namespace,
        package='isaac_ros_hawk',
        plugin='nvidia::isaac_ros::hawk::HawkNode',
        parameters=[{
            'module_id': hawk_module_id,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('correlated_timestamp', '/correlated_timestamp'),
        ]
    )

    composable_node_descriptions = [hawk_node]
    if correlated_timestamp_driver_node:
        composable_node_descriptions.append(correlated_timestamp_driver_node)

    load_nodes = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=composable_node_descriptions,
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
            'create_correlated_timestamp_driver_node',
            description='Create a CorrelatedTimestampDriverNode.',
            default_value='True',
        ),
        DeclareLaunchArgument(
            'hawk_placement',
            default_value='front',
            description='Select Hawk camera from "front", "left", "right", "back"'
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
