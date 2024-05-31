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
Performance test for the Isaac ROS Mobile Segment Anything (SAM) graph.

The graph consists of the following:
- Preprocessors:
    1. Resize, Pad, ImageFormatConverter, ImageToTensor, InterleavedToPlanar, Reshape Nodes:
         turns raw images into appropriately-shaped tensors
    2. RtDetrPreprocessorNode: attaches required image size tensor to encoded image tensor
    3. TensorRTNode: runs RT-DETR inference
    4. RtDetrDecoderNode:  fetches bboxes from output tensor.
- Graph under Test:
    1. ResizeNode & PadNode: resizes and pads the image to get expected dimensions(1024x1024)
    2. ImageFormatConverterNode: converts the image to RGB8 color space
    3. ImageToTensorNode: converts image to tensor
    4. ImageTensorNormalizeNode: normalizes the tensor
    5. InterleavedToPlanarNode: converts tensor from HWC to CHW format
    6. ReshapeNode: converts tensor to NCHW format
    7. SegmentAnythingDataEncoderNode: preprocesses the input prompt data
    8. DummyMaskPublisher: publishes the zero mask for SAM inference
    9. TritonNode: runs the SAM inference
    10. SegmentAnythingDecoderNode: decodes the inference output and returns the masks

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_triton
    - isaac_ros_tensor_rt
    - isaac_ros_rtdetr
    - isaac_ros_segment_anything
- Datasets:
    - assets/datasets/r2b_dataset/r2b_robotarm
- Models:
    - assets/models/sdetr/sdetr_grasp.etlt
    - assets/models/segment_anything/mobile_sam.onnx
    - assets/models/segment_anything/config.pbtxt
"""

import os
import shutil
import time

from isaac_ros_benchmark import TaoConverter
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution, ROS2BenchmarkConfig, ROS2BenchmarkTest

IMAGE_RESOLUTION = ImageResolution.HD
NETWORK_SIZE = 640  # RT-DETR architecture requires square network resolution
RTDETR_NETWORK_RESOLUTION = Resolution(NETWORK_SIZE, NETWORK_SIZE)
SAM_NETWORK_RESOLUTION = Resolution(1024, 1024)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_robotarm'

MODEL_NAME = 'segment_anything'
MODEL_CONFIG_FILE_NAME = 'segment_anything/config.pbtxt'
MODEL_FILE_NAME = 'segment_anything/mobile_sam.onnx'

TRITON_REPO_PATH = ['/tmp/models/mobile_segment_anything']
PROMPT_INPUT_TYPE = 'bbox'
TRITON_MODEL_DIR = '/tmp/models/mobile_segment_anything/segment_anything/1/'
TRITON_CONFIG_PATH = '/tmp/models/mobile_segment_anything/segment_anything/'

RTDETR_MODEL_FILE_NAME = 'sdetr/sdetr_grasp.etlt'
RTDETR_ENGINE_FILE_PATH = '/tmp/sdetr_grasp.plan'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS Mobile Segment Anything (SAM) graph."""
    resize_node = ComposableNode(
        name='SamResizeNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': SAM_NETWORK_RESOLUTION['width'],
            'output_height': SAM_NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'disable_padding': True,
            'input_width': NETWORK_SIZE,
            'input_height': NETWORK_SIZE
        }],
        remappings=[
            ('resize/image', 'segment_anything/resized_image'),
            ('image', 'playback_node/image'),
            ('camera_info', 'playback_node/camera_info')
        ]
    )

    pad_node = ComposableNode(
        name='SamPadNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': SAM_NETWORK_RESOLUTION['width'],
            'output_image_height': SAM_NETWORK_RESOLUTION['height'],
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[('image', 'segment_anything/resized_image'),
                    ('padded_image', 'segment_anything/padded_image')]
    )

    image_format_converter_node = ComposableNode(
        name='ImageFormatConverter',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': SAM_NETWORK_RESOLUTION['width'],
            'image_height': SAM_NETWORK_RESOLUTION['height'],
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'segment_anything/padded_image'),
            ('image', 'segment_anything/color_converted_image'),
        ],
    )

    image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': True,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'segment_anything/color_converted_image'),
            ('tensor', 'segment_anything/image_tensor')
        ]
    )

    normalize_node = ComposableNode(
        name='SamNormalizeNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': [0.485, 0.456, 0.406],
            'stddev': [0.229, 0.224, 0.225],
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }],
        remappings=[
            ('tensor', 'segment_anything/image_tensor'),
            ('normalized_tensor', 'segment_anything/normalized_tensor'),
        ],
    )

    interleaved_to_planar_node = ComposableNode(
        name='SamInterleavedToPlanarNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [SAM_NETWORK_RESOLUTION['height'],
                                   SAM_NETWORK_RESOLUTION['width'], 3]
        }],
        remappings=[
            ('interleaved_tensor', 'segment_anything/normalized_tensor'),
            ('planar_tensor', 'segment_anything/planar_tensor'),
        ]
    )

    reshaper_node = ComposableNode(
        name='ReshapeNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, SAM_NETWORK_RESOLUTION['height'],
                                   SAM_NETWORK_RESOLUTION['width']],
            'output_tensor_shape': [1, 3, SAM_NETWORK_RESOLUTION['height'],
                                    SAM_NETWORK_RESOLUTION['width']],
            'reshaped_tensor_nitros_format': 'nitros_tensor_list_nchw_rgb_f32',
        }],
        remappings=[
            ('tensor', 'segment_anything/planar_tensor'),
            ('reshaped_tensor', 'segment_anything/tensor_pub'),
        ],
    )

    dummy_mask_pub_node = ComposableNode(
        name='DummyMaskPub',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::DummyMaskPublisher',
        remappings=[('tensor_pub', 'segment_anything/tensor_pub')]
    )

    data_preprocessor_node = ComposableNode(
        name='SamDataEncoderNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDataEncoderNode',
        parameters=[{
            'prompt_input_type': 'bbox',
            'has_input_mask': False,
            'max_batch_size': 20,
            'orig_img_dims': [640, 640]
        }],
        remappings=[
            ('prompts', 'playback_node/detections_output'),
            ('tensor_pub', 'segment_anything/tensor_pub'),
            ('tensor', 'segment_anything/encoded_data')
        ]
    )

    sam_triton_node = ComposableNode(
        name='SamTritonNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': TRITON_REPO_PATH,
            'max_batch_size': 1,
            'input_tensor_names': ['input_tensor', 'points', 'labels', 'input_mask',
                                   'has_input_mask', 'orig_img_dims'],
            'input_binding_names': ['images', 'point_coords', 'point_labels', 'mask_input',
                                    'has_mask_input', 'orig_im_size'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['masks', 'iou', 'low_res_mask'],
            'output_binding_names': ['masks', 'iou_predictions', 'low_res_masks'],
            'output_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32']
        }],
        remappings=[
            ('tensor_pub', 'segment_anything/encoded_data'),
            ('tensor_sub', 'segment_anything/tensor_sub')
        ]
    )

    sam_decoder_node = ComposableNode(
        name='SemgnetAnythingDecoderNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
        parameters=[{
            'mask_width': 640,
            'mask_height': 640,
            'max_batch_size': 20
        }],
        remappings=[('tensor_sub', 'segment_anything/tensor_sub')])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info')
        ]
    )

    prep_rtdetr_resize_node = ComposableNode(
        name='PrepRtdetrResizeNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height'],
            'output_width': RTDETR_NETWORK_RESOLUTION['width'],
            'output_height': RTDETR_NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True
        }],
        remappings=[
            ('image', 'data_loader/image_raw'),
            ('camera_info', 'data_loader/camera_info')
        ]
    )

    prep_rtdetr_pad_node = ComposableNode(
        name='PrepRtdetrPadNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': RTDETR_NETWORK_RESOLUTION['width'],
            'output_image_height': RTDETR_NETWORK_RESOLUTION['height'],
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[
            ('image', 'resize/image')
        ]
    )

    prep_rtdetr_image_format_node = ComposableNode(
        name='PrepRtdetrImageFormatNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
                'encoding_desired': 'rgb8',
                'image_width': RTDETR_NETWORK_RESOLUTION['width'],
                'image_height': RTDETR_NETWORK_RESOLUTION['height']
        }],
        remappings=[
            ('image_raw', 'padded_image'),
            ('image', 'image_rgb')]
    )

    prep_rtdetr_image_to_tensor_node = ComposableNode(
        name='PrepRtdetrImageToTensorNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': False,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'image_rgb'),
            ('tensor', 'normalized_tensor'),
        ]
    )

    prep_rtdetr_interleave_to_planar_node = ComposableNode(
        name='PrepRtdetrInterleavedToPlanarNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [RTDETR_NETWORK_RESOLUTION['width'],
                                   RTDETR_NETWORK_RESOLUTION['height'], 3]
        }],
        remappings=[
            ('interleaved_tensor', 'normalized_tensor')
        ]
    )

    prep_rtdetr_reshape_node = ComposableNode(
        name='PrepRtdetrReshapeNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [3, RTDETR_NETWORK_RESOLUTION['width'],
                                   RTDETR_NETWORK_RESOLUTION['height']],
            'output_tensor_shape': [
                1, 3, RTDETR_NETWORK_RESOLUTION['width'], RTDETR_NETWORK_RESOLUTION['height']]
        }],
        remappings=[
            ('tensor', 'planar_tensor')
        ],
    )

    prep_rtdetr_preprocessor_node = ComposableNode(
        name='PrepRtdetrPreprocessor',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        parameters=[{
            'image_size': RTDETR_NETWORK_RESOLUTION['width']
        }],
        remappings=[
            ('encoded_tensor', 'reshaped_tensor')
        ]
    )

    prep_rtdetr_tensor_rt_node = ComposableNode(
        name='PrepRtdetrTensorRt',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'engine_file_path': RTDETR_ENGINE_FILE_PATH,
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'verbose': False,
            'force_engine_update': False
        }]
    )

    prep_rtdetr_decoder_node = ComposableNode(
        name='PrepRtdetrDecoder',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
    )
    # Preprocess graph end #

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_image_rgb8', 'nitros_camera_info', 'nitros_detection2_d_array'
            ],
        }],
        remappings=[
            ('buffer/input0', 'padded_image'),
            ('input0', 'playback_node/image'),
            ('buffer/input1', 'resize/camera_info'),
            ('input1', 'playback_node/camera_info'),
            ('buffer/input2', 'detections_output'),
            ('input2', 'playback_node/detections_output')
        ]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_tensor_list_nchw',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'segment_anything/raw_segmentation_mask')]
    )

    composable_node_container = ComposableNodeContainer(
        name='sam_container',
        namespace=TestIsaacROSSegmentAnythingGraph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
            prep_rtdetr_resize_node,
            prep_rtdetr_pad_node,
            prep_rtdetr_image_format_node,
            prep_rtdetr_image_to_tensor_node,
            prep_rtdetr_reshape_node,
            prep_rtdetr_preprocessor_node,
            prep_rtdetr_tensor_rt_node,
            prep_rtdetr_decoder_node,
            prep_rtdetr_interleave_to_planar_node,
            playback_node,
            resize_node,
            pad_node,
            sam_decoder_node,
            monitor_node,
            image_format_converter_node,
            image_to_tensor_node,
            interleaved_to_planar_node,
            reshaper_node,
            sam_triton_node,
            data_preprocessor_node,
            dummy_mask_pub_node,
            normalize_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSSegmentAnythingGraph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)

    if not os.path.isfile(MODEL_FILE_PATH):
        raise SystemExit('Model file not found.')

    if not os.path.exists(os.path.dirname(TRITON_MODEL_DIR)):
        os.makedirs(os.path.dirname(TRITON_MODEL_DIR))

    shutil.copy(
        os.path.join(MODELS_ROOT, MODEL_CONFIG_FILE_NAME),
        TRITON_CONFIG_PATH)
    shutil.copy(
        MODEL_FILE_PATH,
        os.path.join(TRITON_MODEL_DIR, 'model.onnx'))

    RTDETR_MODEL_FILE_PATH = os.path.join(MODELS_ROOT, RTDETR_MODEL_FILE_NAME)

    # Generate engine file using tao-converter
    if not os.path.isfile(RTDETR_ENGINE_FILE_PATH):
        tao_converter_args = [
            '-k', 'sdetr',
            '-t', 'fp16',
            '-e', RTDETR_ENGINE_FILE_PATH,
            '-p', 'images,1x3x640x640,2x3x640x640,4x3x640x640',
            '-p', 'orig_target_sizes,1x2,2x2,4x2',
            RTDETR_MODEL_FILE_PATH
        ]
        TaoConverter()(tao_converter_args)

    return TestIsaacROSSegmentAnythingGraph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSSegmentAnythingGraph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Mobile Segment Anything (SAM) graph benchmark."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Mobile Segment Anything (SAM) Graph Benchmark',
        input_data_path=ROSBAG_PATH,
        # The slice of the rosbag to use
        input_data_start_time=2.1,
        input_data_end_time=2.2,
        # Upper and lower bounds of peak throughput search window
        publisher_upper_frequency=200.0,
        publisher_lower_frequency=2.0,
        post_trial_run_wait_time_sec=20,
        pre_trial_run_wait_time_sec=5.0,
        # The number of frames to be buffered
        playback_message_buffer_size=1,
        custom_report_info={
            'data_resolution': IMAGE_RESOLUTION,
            'network_resolution': SAM_NETWORK_RESOLUTION
        }
    )

    # Amount of seconds to wait for Triton Engine to be initialized
    TRITON_WAIT_SEC = 10

    def pre_benchmark_hook(self):
        # Wait for model to be generated
        # Note that the model engine file exist only if previous model conversion succeeds.
        # Wait for Triton Node to be launched
        while not os.path.isfile(RTDETR_ENGINE_FILE_PATH):
            time.sleep(1)

        # Wait for Triton Node to be loaded
        time.sleep(self.TRITON_WAIT_SEC)

    def test_benchmark(self):
        self.run_benchmark()
