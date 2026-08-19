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

import isaac_ros_ess_benchmark.ess_model_utility as ess_model_utility

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode

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
    network_width = ess_resolution['width']
    network_height = ess_resolution['height']
    tensor_memory_pool_block_size = network_width * network_height * 3 * 4

    engine_file_path = LaunchConfiguration('engine_file_path').perform(context)
    if LaunchConfigurationEquals('engine_file_path', '').evaluate(context):
        _, engine_file_path = ess_model_utility.get_model_paths(ess_model_type)
    print('\tUse ESS engine file path: {}'.format(engine_file_path))

    engine_parent = os.path.dirname(engine_file_path)
    asset_models_root = None
    if os.path.basename(engine_parent) == 'ess':
        asset_models_root = os.path.dirname(engine_parent)
    ess_plugin_path = ess_model_utility.get_plugin_path(asset_models_root)

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

    left_format_node = ComposableNode(
        name='LeftFormatNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image_raw', 'left/image_rect'),
            ('image', 'left/image_rgb')
        ]
    )

    left_resize_node = ComposableNode(
        name='LeftResizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': network_width,
            'output_height': network_height,
            'keep_aspect_ratio': False,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image', 'left/image_rgb'),
            ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )

    left_normalize_node = ComposableNode(
        name='LeftNormalizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'left/image_resize'),
            ('normalized_image', 'left/image_normalize')
        ]
    )

    left_tensor_node = ComposableNode(
        name='LeftTensorNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'left_image',
            'memory_pool_block_size': tensor_memory_pool_block_size,
        }],
        remappings=[
            ('image', 'left/image_normalize'),
            ('tensor', 'left/tensor')
        ]
    )

    left_planar_node = ComposableNode(
        name='LeftPlanarNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'left_image',
        }],
        remappings=[
            ('interleaved_tensor', 'left/tensor'),
            ('planar_tensor', 'left/tensor_planar')
        ]
    )

    left_reshape_node = ComposableNode(
        name='LeftReshapeNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[
            ('tensor', 'left/tensor_planar'),
            ('reshaped_tensor', 'left/tensor_reshape')
        ]
    )

    right_format_node = ComposableNode(
        name='RightFormatNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image_raw', 'right/image_rect'),
            ('image', 'right/image_rgb')
        ]
    )

    right_resize_node = ComposableNode(
        name='RightResizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': network_width,
            'output_height': network_height,
            'keep_aspect_ratio': False,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('image', 'right/image_rgb'),
            ('camera_info', 'right/camera_info_rect'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )

    right_normalize_node = ComposableNode(
        name='RightNormalizeNode',
        namespace=node_namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{
            'mean': [127.5, 127.5, 127.5],
            'stddev': [127.5, 127.5, 127.5],
        }],
        remappings=[
            ('image', 'right/image_resize'),
            ('normalized_image', 'right/image_normalize')
        ]
    )

    right_tensor_node = ComposableNode(
        name='RightTensorNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'right_image',
            'memory_pool_block_size': tensor_memory_pool_block_size,
        }],
        remappings=[
            ('image', 'right/image_normalize'),
            ('tensor', 'right/tensor')
        ]
    )

    right_planar_node = ComposableNode(
        name='RightPlanarNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'right_image',
        }],
        remappings=[
            ('interleaved_tensor', 'right/tensor'),
            ('planar_tensor', 'right/tensor_planar')
        ]
    )

    right_reshape_node = ComposableNode(
        name='RightReshapeNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[
            ('tensor', 'right/tensor_planar'),
            ('reshaped_tensor', 'right/tensor_reshape')
        ]
    )

    tensor_pair_sync_node = ComposableNode(
        name='TensorPairSyncNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image',
            'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'input_left',
            'output_tensor2_name': 'input_right',
        }],
        remappings=[
            ('tensor1', 'left/tensor_reshape'),
            ('tensor2', 'right/tensor_reshape'),
        ]
    )

    tensor_rt_node = ComposableNode(
        name='TensorRTNode',
        namespace=node_namespace,
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': engine_file_path,
            'input_tensor_names': ['input_left', 'input_right'],
            'input_binding_names': ['input_left', 'input_right'],
            'output_tensor_names': ['output_left', 'output_conf'],
            'output_binding_names': ['output_left', 'output_conf'],
            'verbose': False,
            'force_engine_update': False,
            'custom_plugin_lib': ess_plugin_path,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }]
    )

    dnn_stereo_decoder_node = ComposableNode(
        name='DNNStereoDecoderNode',
        namespace=node_namespace,
        package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'output_left',
            'confidence_tensor_name': 'output_conf',
            'confidence_threshold': 0.4,
            'cache_camera_info': True,
            'reusable_buffer_enable': False,
            'type_negotiation_duration_s': type_negotiation_duration_s,
        }],
        remappings=[
            ('right/camera_info', 'right/camera_info_resize')
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
            left_format_node,
            left_resize_node,
            left_normalize_node,
            left_tensor_node,
            left_planar_node,
            left_reshape_node,
            right_format_node,
            right_resize_node,
            right_normalize_node,
            right_tensor_node,
            right_planar_node,
            right_reshape_node,
            tensor_pair_sync_node,
            tensor_rt_node,
            dnn_stereo_decoder_node,
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
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
