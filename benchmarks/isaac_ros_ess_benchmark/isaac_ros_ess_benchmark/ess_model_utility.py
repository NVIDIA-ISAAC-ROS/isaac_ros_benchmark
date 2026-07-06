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
import time

from isaac_ros_ess.engine_generator import ESSEngineGenerator
from ros2_benchmark import ROS2BenchmarkTest

# ESS mode paths
FULL_ESS_MODEL_FILE_NAME = 'ess/ess.onnx'
FULL_ESS_ENGINE_FILE_PATH = 'ess/ess.engine'
LIGHT_ESS_MODEL_FILE_NAME = 'ess/light_ess.onnx'
LIGHT_ESS_ENGINE_FILE_PATH = 'ess/light_ess.engine'

# ESS mode resolution
FULL_ESS_RESOLUTION = {'width': 960, 'height': 576}
LIGHT_ESS_RESOLUTION = {'width': 480, 'height': 288}


def get_mode_resolution(ess_model_type):
    if ess_model_type == 'full':
        return FULL_ESS_RESOLUTION
    elif ess_model_type == 'light':
        return LIGHT_ESS_RESOLUTION
    else:
        raise ValueError('Unrecognized ESS model type: {}'.format(ess_model_type))


def get_model_root(asset_models_root=None):
    if asset_models_root is None:
        return os.path.join(ROS2BenchmarkTest.get_assets_root_path(), 'models')
    else:
        return asset_models_root


def get_model_paths(ess_model_type, asset_models_root=None):
    model_root = get_model_root(asset_models_root)

    if ess_model_type == 'full':
        ess_model_path = os.path.join(model_root, FULL_ESS_MODEL_FILE_NAME)
        ess_engine_path = os.path.join(model_root, FULL_ESS_ENGINE_FILE_PATH)
    elif ess_model_type == 'light':
        ess_model_path = os.path.join(model_root, LIGHT_ESS_MODEL_FILE_NAME)
        ess_engine_path = os.path.join(model_root, LIGHT_ESS_ENGINE_FILE_PATH)
    else:
        raise ValueError('Unrecognized ESS model type: {}'.format(ess_model_type))

    return ess_model_path, ess_engine_path


def generate_ess_engine_file(ess_model_type, asset_models_root=None):
    ess_model_path, ess_engine_path = get_model_paths(ess_model_type, asset_models_root)

    # Generate engine file using trtexec
    if not os.path.isfile(ess_engine_path):
        print(f'Generating engine file for {ess_model_type} ESS model...')
        gen = ESSEngineGenerator(onnx_model=ess_model_path)
        start_time = time.time()
        gen.generate()
        print('ESS model engine file generation was finished '
              f'(took {(time.time() - start_time)}s)')
    else:
        print(f'An ESS engine file was found at "{ess_engine_path}"')


def is_ess_engine_file_generated(ess_model_type='full', asset_models_root=None):
    _, ess_engine_path = get_model_paths(ess_model_type, asset_models_root)
    return os.path.isfile(ess_engine_path)


def get_plugin_path(asset_models_root=None):
    """Get the ESS TensorRT plugin library path."""
    import platform
    arch = platform.machine()
    model_root = get_model_root(asset_models_root)
    plugin_path = os.path.join(model_root, 'ess', 'plugins', arch, 'ess_plugins.so')
    if os.path.isfile(plugin_path):
        return plugin_path
    return ''


def create_ess_pipeline_nodes(namespace, network_width, network_height, engine_file_path,
                              ess_plugin_path='', threshold=0.4):
    """Create the composable nodes for the full ESS preprocessing + inference + decode pipeline."""
    from launch_ros.descriptions import ComposableNode

    left_format_node = ComposableNode(
        name='left_format_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{'encoding_desired': 'rgb8'}],
        remappings=[('image_raw', 'left/image_rect'), ('image', 'left/image_rgb')]
    )
    left_resize_node = ComposableNode(
        name='left_resize_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': network_width, 'output_height': network_height,
            'keep_aspect_ratio': False,
        }],
        remappings=[
            ('image', 'left/image_rgb'), ('camera_info', 'left/camera_info_rect'),
            ('resize/image', 'left/image_resize'),
            ('resize/camera_info', 'left/camera_info_resize'),
        ]
    )
    left_normalize_node = ComposableNode(
        name='left_normalize_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{'mean': [127.5, 127.5, 127.5], 'stddev': [127.5, 127.5, 127.5]}],
        remappings=[('image', 'left/image_resize'), ('normalized_image', 'left/image_normalize')]
    )
    left_tensor_node = ComposableNode(
        name='left_tensor_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'left_image'}],
        remappings=[('image', 'left/image_normalize'), ('tensor', 'left/tensor')]
    )
    left_planar_node = ComposableNode(
        name='left_planar_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'left_image',
        }],
        remappings=[('interleaved_tensor', 'left/tensor'), ('planar_tensor', 'left/tensor_planar')]
    )
    left_reshape_node = ComposableNode(
        name='left_reshape_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'left_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[('tensor', 'left/tensor_planar'), ('reshaped_tensor', 'left/tensor_reshape')]
    )

    right_format_node = ComposableNode(
        name='right_format_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{'encoding_desired': 'rgb8'}],
        remappings=[('image_raw', 'right/image_rect'), ('image', 'right/image_rgb')]
    )
    right_resize_node = ComposableNode(
        name='right_resize_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': network_width, 'output_height': network_height,
            'keep_aspect_ratio': False,
        }],
        remappings=[
            ('image', 'right/image_rgb'), ('camera_info', 'right/camera_info_rect'),
            ('resize/image', 'right/image_resize'),
            ('resize/camera_info', 'right/camera_info_resize'),
        ]
    )
    right_normalize_node = ComposableNode(
        name='right_normalize_node', namespace=namespace,
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageNormalizeNode',
        parameters=[{'mean': [127.5, 127.5, 127.5], 'stddev': [127.5, 127.5, 127.5]}],
        remappings=[('image', 'right/image_resize'), ('normalized_image', 'right/image_normalize')]
    )
    right_tensor_node = ComposableNode(
        name='right_tensor_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{'scale': False, 'tensor_name': 'right_image'}],
        remappings=[('image', 'right/image_normalize'), ('tensor', 'right/tensor')]
    )
    right_planar_node = ComposableNode(
        name='right_planar_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [network_height, network_width, 3],
            'output_tensor_name': 'right_image',
        }],
        remappings=[
            ('interleaved_tensor', 'right/tensor'), ('planar_tensor', 'right/tensor_planar')
        ]
    )
    right_reshape_node = ComposableNode(
        name='right_reshape_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'right_image',
            'input_tensor_shape': [3, network_height, network_width],
            'output_tensor_shape': [1, 3, network_height, network_width],
        }],
        remappings=[('tensor', 'right/tensor_planar'), ('reshaped_tensor', 'right/tensor_reshape')]
    )

    tensor_pair_sync_node = ComposableNode(
        name='tensor_pair_sync_node', namespace=namespace,
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::TensorPairSyncNode',
        parameters=[{
            'input_tensor1_name': 'left_image', 'input_tensor2_name': 'right_image',
            'output_tensor1_name': 'input_left', 'output_tensor2_name': 'input_right',
        }],
        remappings=[('tensor1', 'left/tensor_reshape'), ('tensor2', 'right/tensor_reshape')]
    )
    tensor_rt_node = ComposableNode(
        name='tensor_rt', namespace=namespace,
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
        }]
    )
    dnn_stereo_decoder_node = ComposableNode(
        name='dnn_stereo_decoder', namespace=namespace,
        package='isaac_ros_dnn_stereo_decoder',
        plugin='nvidia::isaac_ros::dnn_stereo_depth::DNNStereoDecoderNode',
        parameters=[{
            'disparity_tensor_name': 'output_left',
            'confidence_tensor_name': 'output_conf',
            'confidence_threshold': threshold,
            'cache_camera_info': True,
            'reusable_buffer_enable': False,
        }],
        remappings=[('right/camera_info', 'right/camera_info_resize')]
    )

    return [
        left_format_node, left_resize_node, left_normalize_node,
        left_tensor_node, left_planar_node, left_reshape_node,
        right_format_node, right_resize_node, right_normalize_node,
        right_tensor_node, right_planar_node, right_reshape_node,
        tensor_pair_sync_node, tensor_rt_node, dnn_stereo_decoder_node,
    ]
