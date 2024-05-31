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
"""
Performance test for the Isaac FoundationPose Tracking Node.

The graph consists of the following:
- Preprocessors:
    1. RtDetr Graph: DNN image encoder and RT-DETR inference for segmentation
    2. ConvertMetricNode: Converts uint16/millimeters to float32/meters
    3. FoundationPose Node: Initial pose estimation
- Graph under Test:
    1. FoundationPose Tracking Node: detects 6D Pose of objects

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_stereo_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_tensor_rt
    - isaac_ros_ess
    - isaac_ros_rtdetr
    - isaac_ros_foundationpose
- Datasets:
    - assets/datasets/r2b_dataset/r2b_robotarm
- Models:
    - assets/models/sdetr/sdetr_grasp.etlt
    - assets/models/ess/ess.etlt
    - assets/models/foundationpose/refine_model.etlt
    - assets/models/foundationpose/score_model.etlt
- Configs:
    - assets/configs/Mac_and_cheese_0_1/Mac_and_cheese_0_1.obj
    - assets/configs/Mac_and_cheese_0_1/materials/textures/baked_mesh_tex0.png
"""

import os
import time

from isaac_ros_benchmark import TaoConverter

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution
from ros2_benchmark import ROS2BenchmarkConfig, ROS2BenchmarkTest

ROSBAG_PATH = 'datasets/r2b_dataset/r2b_robotarm'

MESH_FILE_NAME = 'Mac_and_cheese_0_1/Mac_and_cheese_0_1.obj'
TEXTURE_MAP_NAME = 'Mac_and_cheese_0_1/materials/textures/baked_mesh_tex0.png'

REFINE_MODEL_NAME = 'foundationpose/refine_model.etlt'
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'

SCORE_MODEL_NAME = 'foundationpose/score_model.etlt'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'

RT_DETR_MODEL_NAME = 'sdetr/sdetr_grasp.etlt'
RT_DETR_ENGINE_PATH = '/tmp/sdetr_grasp.plan'

IMAGE_RESOLUTION = ImageResolution.HD

# RT-DETR models expect 640x640 encoded image size
RT_DETR_MODEL_INPUT_SIZE = 640
# RT-DETR models expect 3 image channels
RT_DETR_MODEL_NUM_CHANNELS = 3

INPUT_TO_RT_DETR_RATIO = IMAGE_RESOLUTION['width'] / RT_DETR_MODEL_INPUT_SIZE


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for benchmarking Isaac ROS FoundationPose graph."""
    config_root = os.path.join(TestIsaacROSFoundationPoseGraph.get_assets_root_path(), 'configs')
    MESH_FILE_PATH = os.path.join(config_root, MESH_FILE_NAME)
    TEXTURE_MAP_PATH = os.path.join(config_root, TEXTURE_MAP_NAME)

    foundationpose_tracking_node = ComposableNode(
        name='FoundationposeTrackingNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
        parameters=[{
            'mesh_file_path': MESH_FILE_PATH,
            'texture_path': TEXTURE_MAP_PATH,
            'refine_engine_file_path': REFINE_ENGINE_PATH,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
        }]
    )

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info'),
            ('camera_1/aligned_depth_to_color/image_raw',
             'data_loader/aligned_depth_to_color/image_raw'),
            ('/camera_1/aligned_depth_to_color/camera_info',
             'data_loader/aligned_depth_to_color/camera_info'),
        ]
    )

    prep_convert_metric_node = ComposableNode(
        name='ConvertMetricNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_depth_image_proc',
        plugin='nvidia::isaac_ros::depth_image_proc::ConvertMetricNode',
        remappings=[
            ('image_raw', 'data_loader/aligned_depth_to_color/image_raw'),
            ('image', 'depth_image_32fc1')
        ]
    )

    prep_resize_left_rt_detr_node = ComposableNode(
        name='ResizeLeftRTDetrNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height'],
            'output_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_height': RT_DETR_MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'data_loader/image_raw'),
            ('camera_info', 'data_loader/camera_info'),
            ('resize/image', 'color_image_resized'),
            ('resize/camera_info', 'camera_info_resized')
        ]
    )

    # Pad the image from IMAGE_WIDTH/INPUT_TO_RT_DETR_RATIO x
    # IMAGE_HEIGHT/INPUT_TO_RT_DETR_RATIO
    # to RT_DETR_MODEL_INPUT_WIDTH x RT_DETR_MODEL_INPUT_HEIGHT
    prep_pad_node = ComposableNode(
        name='PadNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RT_DETR_MODEL_INPUT_SIZE,
            'output_image_height': RT_DETR_MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'color_image_resized')]
    )

    # Convert image to tensor and reshape
    prep_image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'padded_image'),
            ('tensor', 'normalized_tensor'),
        ]
    )

    prep_interleave_to_planar_node = ComposableNode(
        name='InterleavedToPlanarNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [
                RT_DETR_MODEL_INPUT_SIZE, RT_DETR_MODEL_INPUT_SIZE, RT_DETR_MODEL_NUM_CHANNELS
            ]
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')])

    prep_reshape_node = ComposableNode(
        name='ReshapeNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [
                RT_DETR_MODEL_NUM_CHANNELS, RT_DETR_MODEL_INPUT_SIZE, RT_DETR_MODEL_INPUT_SIZE
            ],
            'output_tensor_shape': [
                1, RT_DETR_MODEL_NUM_CHANNELS, RT_DETR_MODEL_INPUT_SIZE, RT_DETR_MODEL_INPUT_SIZE
            ]
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    prep_rtdetr_preprocessor_node = ComposableNode(
        name='RTDetrPreprocessorNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[('encoded_tensor', 'reshaped_tensor')])

    # RT-DETR objection detection pipeline
    prep_rtdetr_tensor_rt_node = ComposableNode(
        name='RTDetrTensorRTNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': RT_DETR_ENGINE_PATH,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'force_engine_update': False
        }]
    )

    prep_rtdetr_decoder_node = ComposableNode(
        name='RTDetrDecoderNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
    )

    # Create a binary segmentation mask from a Detection2DArray published by RT-DETR.
    # The segmentation mask is of size
    # int(IMAGE_WIDTH/INPUT_TO_RT_DETR_RATIO) x int(IMAGE_HEIGHT/INPUT_TO_RT_DETR_RATIO)
    prep_detection2d_to_mask_node = ComposableNode(
        name='Detection2dToMaskNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': int(IMAGE_RESOLUTION['width'] / INPUT_TO_RT_DETR_RATIO),
            'mask_height': int(IMAGE_RESOLUTION['height'] / INPUT_TO_RT_DETR_RATIO)
        }],
        remappings=[
            ('detection2_d_array', 'detections_output'),
            ('segmentation', 'rt_detr_segmentation')
        ]
    )

    # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
    # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
    # Resize from int(IMAGE_WIDTH/INPUT_TO_RT_DETR_RATIO) x
    # int(IMAGE_HEIGHT/INPUT_TO_RT_DETR_RATIO)
    # to depth width x depth height
    # output height constraint is used since keep_aspect_ratio is False
    # and the image is padded
    prep_resize_mask_node = ComposableNode(
        name='ResizeMaskNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(IMAGE_RESOLUTION['width'] / INPUT_TO_RT_DETR_RATIO),
            'input_height': int(IMAGE_RESOLUTION['height'] / INPUT_TO_RT_DETR_RATIO),
            'output_width': IMAGE_RESOLUTION['width'],
            'output_height': IMAGE_RESOLUTION['height'],
            'keep_aspect_ratio': False,
            'disable_padding': False
        }],
        remappings=[
            ('image', 'rt_detr_segmentation'),
            ('camera_info', 'camera_info_resized'),
            ('resize/image', 'prep/segmentation'),
            ('resize/camera_info', 'prep/camera_info_segmentation')
        ]
    )

    prep_foundationpose_node = ComposableNode(
        name='FoundationPoseNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': MESH_FILE_PATH,
            'texture_path': TEXTURE_MAP_PATH,
            'refine_engine_file_path': REFINE_ENGINE_PATH,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],
            'score_engine_file_path': SCORE_ENGINE_PATH,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        remappings=[
            ('pose_estimation/depth_image', 'depth_image_32fc1'),
            ('pose_estimation/image', 'data_loader/image_raw'),
            ('pose_estimation/camera_info', 'data_loader/camera_info'),
            ('pose_estimation/segmentation', 'prep/segmentation'),
            ('pose_estimation/pose_matrix_output', 'buffer/tracking/pose_matrix_output')
        ]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_image_32FC1', 'nitros_image_rgb8', 'nitros_camera_info',
                'nitros_tensor_list_nchw'
            ],
        }],
        remappings=[
            ('buffer/input0', 'depth_image_32fc1'),
            ('input0', 'tracking/depth_image'),
            ('buffer/input1', 'data_loader/image_raw'),
            ('input1', 'tracking/image'),
            ('buffer/input2', 'data_loader/camera_info'),
            ('input2', 'tracking/camera_info'),
            ('buffer/input3', 'buffer/tracking/pose_matrix_output'),
            ('input3', 'tracking/pose_input')
        ],
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_detection3_d_array',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[('output', 'tracking/output')],
    )

    composable_node_container = ComposableNodeContainer(
        name='container',
        namespace=TestIsaacROSFoundationPoseGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_convert_metric_node,
            prep_resize_left_rt_detr_node,
            prep_image_to_tensor_node, prep_pad_node, prep_reshape_node,
            prep_interleave_to_planar_node, prep_rtdetr_preprocessor_node,
            prep_rtdetr_tensor_rt_node, prep_rtdetr_decoder_node, prep_detection2d_to_mask_node,
            prep_resize_mask_node, prep_foundationpose_node,
            playback_node,
            foundationpose_tracking_node,
            monitor_node,
        ],
        output='screen')

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSFoundationPoseGraph.get_assets_root_path(), 'models')
    RT_DETR_MODEL_PATH = os.path.join(MODELS_ROOT, RT_DETR_MODEL_NAME)
    REFINE_MODEL_PATH = os.path.join(MODELS_ROOT, REFINE_MODEL_NAME)
    SCORE_MODEL_PATH = os.path.join(MODELS_ROOT, SCORE_MODEL_NAME)

    # Generate engine file using tao-converter
    if not os.path.isfile(RT_DETR_ENGINE_PATH):
        print('Generating an engine file for the RT-DETR model...')
        start_time = time.time()
        tao_converter_args = [
            '-k', 'sdetr',
            '-t', 'fp16',
            '-e', RT_DETR_ENGINE_PATH,
            '-p', 'images,1x3x640x640,2x3x640x640,4x3x640x640',
            '-p', 'orig_target_sizes,1x2,2x2,4x2',
            RT_DETR_MODEL_PATH
        ]
        TaoConverter()(tao_converter_args)
        print('RT-DETR model engine file generation was finished '
              f'(took {(time.time() - start_time)}s)')

    if not os.path.isfile(REFINE_ENGINE_PATH):
        print('Generating an engine file for the Refine model...')
        start_time = time.time()
        tao_converter_args = [
            '-k', 'foundationpose',
            '-t', 'fp16',
            '-e', REFINE_ENGINE_PATH,
            '-p', 'input1,1x160x160x6,1x160x160x6,252x160x160x6',
            '-p', 'input2,1x160x160x6,1x160x160x6,252x160x160x6',
            '-o', 'output1,output2',
            REFINE_MODEL_PATH
        ]
        TaoConverter()(tao_converter_args)
        print('Refine model engine file generation was finished '
              f'(took {(time.time() - start_time)}s)')

    if not os.path.isfile(SCORE_ENGINE_PATH):
        print('Generating an engine file for the Score model...')
        start_time = time.time()
        tao_converter_args = [
            '-k', 'foundationpose',
            '-t', 'fp16',
            '-e', SCORE_ENGINE_PATH,
            '-p', 'input1,1x160x160x6,1x160x160x6,252x160x160x6',
            '-p', 'input2,1x160x160x6,1x160x160x6,252x160x160x6',
            '-o', 'output1',
            SCORE_MODEL_PATH
        ]
        TaoConverter()(tao_converter_args)
        print('Score model engine file generation was finished '
              f'(took {(time.time() - start_time)}s)')

    return TestIsaacROSFoundationPoseGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSFoundationPoseGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac FoundationPose graph."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac FoundationPose Tracking Node Benchmark',
        input_data_path=ROSBAG_PATH,
        # The slice of the rosbag to use
        input_data_start_time=2.1,
        input_data_end_time=2.2,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=300.0,
        publisher_lower_frequency=1.0,
        benchmark_duration=5,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={'data_resolution': IMAGE_RESOLUTION})

    # Amount of seconds to wait for TensorRT Engine to be initialized
    ENGINE_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Note that if the model is failed to be converted, an exception will be raised and
        # the entire test will end.
        while not os.path.isfile(SCORE_ENGINE_PATH):
            print('Waiting for engine files to be generated')
            time.sleep(1)
        time.sleep(self.ENGINE_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
