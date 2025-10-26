# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Performance test for the Isaac ROS Segment Anything2 (SAM2) graph.

The graph consists of the following:
- Graph under Test:
    1. ResizeNode & PadNode: resizes and pads the image to get expected dimensions(1024x1024)
    2. ImageFormatConverterNode: converts the image to RGB8 color space
    3. ImageToTensorNode: converts image to tensor
    4. ImageTensorNormalizeNode: normalizes the tensor
    5. InterleavedToPlanarNode: converts tensor from HWC to CHW format
    6. ReshapeNode: converts tensor to NCHW format
    7. SegmentAnything2DataEncoderNode: Encodes the input image along with tracked objects data.
    9. TritonNode: runs the SAM2 inference
    10. SegmentAnythingDecoderNode: decodes the inference output and returns the masks

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_tensor_proc
    - isaac_ros_triton
    - isaac_ros_tensor_rt
    - isaac_ros_segment_anything
    - isaac_ros_segment_anything2
    - isaac_ros_segment_anything2_interfaces
- Datasets:
    - assets/datasets/r2b_dataset/r2b_robotarm
- Models:
    - assets/models/segment_anything2/sam2.onnx
    - assets/models/segment_anything2/config.pbtxt
"""

import os
import shutil
import time

# Import service types for adding objects
from isaac_ros_segment_anything2_interfaces.srv import AddObjects
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from ros2_benchmark import ImageResolution, Resolution, ROS2BenchmarkConfig, ROS2BenchmarkTest

from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D

IMAGE_RESOLUTION = ImageResolution.HD
SAM_NETWORK_RESOLUTION = Resolution(1024, 1024)
ROSBAG_PATH = 'datasets/r2b_dataset/r2b_robotarm'

MODEL_NAME = 'segment_anything2'
MODEL_CONFIG_FILE_NAME = 'segment_anything2/config.pbtxt'
MODEL_FILE_NAME = 'segment_anything2/sam2.onnx'

TRITON_REPO_PATH = ['/tmp/models/']
TRITON_MODEL_DIR = '/tmp/models/segment_anything2/1/'
TRITON_CONFIG_PATH = '/tmp/models/segment_anything2/'


def launch_setup(container_prefix, container_sigterm_timeout):
    """Generate launch description for Isaac ROS Segment Anything2 (SAM2) graph."""
    resize_node = ComposableNode(
        name='ResizeNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'output_width': SAM_NETWORK_RESOLUTION['width'],
            'output_height': SAM_NETWORK_RESOLUTION['height'],
            'keep_aspect_ratio': True,
            'disable_padding': True,
            'input_width': IMAGE_RESOLUTION['width'],
            'input_height': IMAGE_RESOLUTION['height']
        }],
        remappings=[
            ('resize/image', 'segment_anything2/resized_image'),
            ('image', 'playback_node/image'),
            ('camera_info', 'playback_node/camera_info')
        ]
    )

    pad_node = ComposableNode(
        name='PadNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': SAM_NETWORK_RESOLUTION['width'],
            'output_image_height': SAM_NETWORK_RESOLUTION['height'],
            'padding_type': 'BOTTOM_RIGHT'
        }],
        remappings=[
            ('image', 'segment_anything2/resized_image'),
            ('padded_image', 'segment_anything2/padded_image')
        ]
    )

    image_format_converter_node = ComposableNode(
        name='ImageFormatConverter',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'image_width': SAM_NETWORK_RESOLUTION['width'],
            'image_height': SAM_NETWORK_RESOLUTION['height'],
            'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'segment_anything2/padded_image'),
            ('image', 'segment_anything2/color_converted_image'),
        ],
    )

    image_to_tensor_node = ComposableNode(
        name='ImageToTensorNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': True,
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'segment_anything2/color_converted_image'),
            ('tensor', 'segment_anything2/image_tensor')
        ]
    )

    normalize_node = ComposableNode(
        name='NormalizeNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageTensorNormalizeNode',
        parameters=[{
            'mean': [0.485, 0.456, 0.406],
            'stddev': [0.229, 0.224, 0.225],
            'input_tensor_name': 'image',
            'output_tensor_name': 'image'
        }],
        remappings=[
            ('tensor', 'segment_anything2/image_tensor'),
            ('normalized_tensor', 'segment_anything2/normalized_tensor'),
        ],
    )

    interleaved_to_planar_node = ComposableNode(
        name='InterleavedToPlanarNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [SAM_NETWORK_RESOLUTION['height'],
                                   SAM_NETWORK_RESOLUTION['width'], 3]
        }],
        remappings=[
            ('interleaved_tensor', 'segment_anything2/normalized_tensor'),
            ('planar_tensor', 'segment_anything2/planar_tensor'),
        ]
    )

    reshaper_node = ComposableNode(
        name='ReshapeNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
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
            ('tensor', 'segment_anything2/planar_tensor'),
            ('reshaped_tensor', 'segment_anything2/tensor_pub'),
        ],
    )

    data_preprocessor_node = ComposableNode(
        name='SegmentAnything2DataEncoderNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_segment_anything2',
        plugin='nvidia::isaac_ros::segment_anything2::SegmentAnything2DataEncoderNode',
        parameters=[{
            'max_num_objects': 5,
            'orig_img_dims': [IMAGE_RESOLUTION['height'], IMAGE_RESOLUTION['width']]
        }],
        remappings=[
            ('encoded_data', 'segment_anything2/encoded_data'),
            ('image', 'segment_anything2/tensor_pub'),
            ('memory', 'segment_anything2/tensor_sub')
        ]
    )

    sam_triton_node = ComposableNode(
        name='TritonNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_triton',
        plugin='nvidia::isaac_ros::dnn_inference::TritonNode',
        parameters=[{
            'model_name': MODEL_NAME,
            'model_repository_paths': TRITON_REPO_PATH,
            'max_batch_size': 1,
            'input_tensor_names': ['image', 'bbox_coords', 'point_coords', 'point_labels',
                                   'mask_memory', 'obj_ptr_memory', 'original_size',
                                   'permutation'],
            'input_binding_names': ['image', 'bbox_coords', 'point_coords', 'point_labels',
                                    'mask_memory', 'obj_ptr_memory', 'original_size',
                                    'permutation'],
            'input_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
            'output_tensor_names': ['high_res_masks', 'object_score_logits',
                                    'maskmem_features', 'maskmem_pos_enc', 'obj_ptr_features'],
            'output_binding_names': ['high_res_masks', 'object_score_logits',
                                     'maskmem_features', 'maskmem_pos_enc', 'obj_ptr_features'],
            'output_tensor_formats': ['nitros_tensor_list_nchw_rgb_f32'],
        }],
        remappings=[
            ('tensor_pub', 'segment_anything2/encoded_data'),
            ('tensor_sub', 'segment_anything2/tensor_sub')
        ]
    )

    sam_decoder_node = ComposableNode(
        name='SemgnetAnythingDecoderNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_segment_anything',
        plugin='nvidia::isaac_ros::segment_anything::SegmentAnythingDecoderNode',
        parameters=[{
            'mask_width': SAM_NETWORK_RESOLUTION['width'],
            'mask_height': SAM_NETWORK_RESOLUTION['height'],
            'max_batch_size': 5,
            'tensor_name': 'high_res_masks'
        }],
        remappings=[
            ('tensor_sub', 'segment_anything2/tensor_sub'),
            ('segment_anything/raw_segmentation_mask', 'segment_anything2/raw_segmentation_mask')
            ])

    data_loader_node = ComposableNode(
        name='DataLoaderNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='ros2_benchmark',
        plugin='ros2_benchmark::DataLoaderNode',
        remappings=[
            ('camera_1/color/image_raw', 'data_loader/image_raw'),
            ('camera_1/color/camera_info', 'data_loader/camera_info')
        ]
    )

    playback_node = ComposableNode(
        name='PlaybackNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosPlaybackNode',
        parameters=[{
            'data_formats': [
                'nitros_image_rgb8', 'nitros_camera_info'
            ],
        }],
        remappings=[
            ('buffer/input0', 'data_loader/image_raw'),
            ('input0', 'playback_node/image'),
            ('buffer/input1', 'data_loader/camera_info'),
            ('input1', 'playback_node/camera_info')
        ]
    )

    monitor_node = ComposableNode(
        name='MonitorNode',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='isaac_ros_benchmark',
        plugin='isaac_ros_benchmark::NitrosMonitorNode',
        parameters=[{
            'monitor_data_format': 'nitros_tensor_list_nchw',
            'use_nitros_type_monitor_sub': True,
        }],
        remappings=[
            ('output', 'segment_anything2/raw_segmentation_mask')
        ]
    )

    composable_node_container = ComposableNodeContainer(
        name='sam_container',
        namespace=TestIsaacROSSegmentAnything2Graph.generate_namespace(),
        package='rclcpp_components',
        executable='component_container_mt',
        prefix=container_prefix,
        sigterm_timeout=container_sigterm_timeout,
        composable_node_descriptions=[
            data_loader_node,
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
            normalize_node
        ],
        output='screen'
    )

    return [composable_node_container]


def generate_test_description():
    MODELS_ROOT = os.path.join(TestIsaacROSSegmentAnything2Graph.get_assets_root_path(), 'models')
    MODEL_FILE_PATH = os.path.join(MODELS_ROOT, MODEL_FILE_NAME)
    print(MODEL_FILE_PATH)
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

    return TestIsaacROSSegmentAnything2Graph.generate_test_description_with_nsys(launch_setup)


class TestIsaacROSSegmentAnything2Graph(ROS2BenchmarkTest):
    """Performance test for the Isaac ROS Segment Anything2 (SAM2) graph benchmark."""

    # Custom configurations
    config = ROS2BenchmarkConfig(
        benchmark_name='Isaac ROS Segment Anything2 (SAM2) Graph Benchmark',
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

    TRITON_WAIT_SEC = 20

    def pre_benchmark_hook(self):
        # Wait for Triton Node to be loaded
        self.get_logger().info('Waiting for Triton Node to initialize...')
        time.sleep(self.TRITON_WAIT_SEC)

        # Add objects before starting the benchmark
        self.get_logger().info('Adding objects for segment_anything2 processing...')
        self.add_objects_for_tracking()

    def add_objects_for_tracking(self):
        """Add objects using the AddObjects service before benchmark execution."""
        # Create service client for AddObjects
        add_objects_client = self.create_service_client_blocking(
            AddObjects, 'add_objects')
        # Select a region that encompasses the right cup from the shelf
        bbox_ids = ['cup']
        bbox = BoundingBox2D()
        bbox.center.position.x = 847.5
        bbox.center.position.y = 70.5
        bbox.size_x = 47.5
        bbox.size_y = 51.5
        bboxes = [bbox]
        # Create the service request
        request = AddObjects.Request()
        # Set header with current time
        request.request_header = Header()
        request.request_header.frame_id = 'camera_frame'

        # Set bbox data
        request.bbox_object_ids = bbox_ids
        request.bbox_coords = bboxes
        self.get_logger().info(f'Adding {len(request.bbox_object_ids)} bboxes')

        # Call the service synchronously
        try:
            add_objects_future = add_objects_client.call_async(request)
            response = self.get_service_response_from_future_blocking(
                add_objects_future, check_success=True)

            self.get_logger().info(f'Successfully added objects: {response.message}')
            self.get_logger().info(f'Object IDs: {response.object_ids}')
            self.get_logger().info(f'Object Indices: {response.object_indices}')

        except Exception as e:
            self.get_logger().error(f'Failed to add objects: {str(e)}')
            raise e

    def test_benchmark(self):
        self.run_benchmark()
