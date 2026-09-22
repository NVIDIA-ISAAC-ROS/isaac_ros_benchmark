// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_benchmark/rosidl_buffer_topic_adapter.hpp"

#include "isaac_ros_pointcloud_interfaces/msg/flat_scan.hpp"
#include "isaac_ros_tensor_msgs/msg/tensor_list.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"

namespace isaac_ros_benchmark
{

ISAAC_ROS_DECLARE_SINGLE_BUFFER_TOPIC_ADAPTERS(
  Image, sensor_msgs::msg::Image, data, header);
ISAAC_ROS_DECLARE_SINGLE_BUFFER_TOPIC_ADAPTERS(
  CompressedImage, sensor_msgs::msg::CompressedImage, data, header);
ISAAC_ROS_DECLARE_SINGLE_BUFFER_TOPIC_ADAPTERS(
  PointCloud2, sensor_msgs::msg::PointCloud2, data, header);

template<>
struct RosidlBufferMessageTraits<stereo_msgs::msg::DisparityImage>
{
  static constexpr bool kHasBufferBackend = true;

  static std::string type_name()
  {
    return rosidl_generator_traits::name<stereo_msgs::msg::DisparityImage>();
  }

  static std::shared_ptr<stereo_msgs::msg::DisparityImage> prepare(
    stereo_msgs::msg::DisparityImage source,
    cudaStream_t stream,
    const rclcpp::Logger & logger)
  {
    const cudaError_t error = clone_buffer_to_cuda(source.image.data, stream);
    return finish_buffering(std::move(source), stream, logger, error);
  }

  static std_msgs::msg::Header & header(stereo_msgs::msg::DisparityImage & message)
  {
    return message.header;
  }

  static const std_msgs::msg::Header & header(
    const stereo_msgs::msg::DisparityImage & message)
  {
    return message.header;
  }
};
ISAAC_ROS_DECLARE_BUFFER_TOPIC_ADAPTERS(
  DisparityImage, stereo_msgs::msg::DisparityImage);

template<>
struct RosidlBufferMessageTraits<isaac_ros_tensor_msgs::msg::TensorList>
{
  static constexpr bool kHasBufferBackend = true;

  static std::string type_name()
  {
    return rosidl_generator_traits::name<
      isaac_ros_tensor_msgs::msg::TensorList>();
  }

  static std::shared_ptr<isaac_ros_tensor_msgs::msg::TensorList> prepare(
    isaac_ros_tensor_msgs::msg::TensorList source,
    cudaStream_t stream,
    const rclcpp::Logger & logger)
  {
    cudaError_t error = cudaSuccess;
    for (auto & tensor : source.tensors) {
      if (error == cudaSuccess) {
        error = clone_buffer_to_cuda(tensor.data, stream);
      }
    }
    return finish_buffering(std::move(source), stream, logger, error);
  }

  static std_msgs::msg::Header & header(
    isaac_ros_tensor_msgs::msg::TensorList & message)
  {
    return message.header;
  }

  static const std_msgs::msg::Header & header(
    const isaac_ros_tensor_msgs::msg::TensorList & message)
  {
    return message.header;
  }
};
ISAAC_ROS_DECLARE_BUFFER_TOPIC_ADAPTERS(
  TensorList, isaac_ros_tensor_msgs::msg::TensorList);

template<>
struct RosidlBufferMessageTraits<isaac_ros_pointcloud_interfaces::msg::FlatScan>
{
  static constexpr bool kHasBufferBackend = false;

  static std::string type_name()
  {
    return rosidl_generator_traits::name<
      isaac_ros_pointcloud_interfaces::msg::FlatScan>();
  }

  static std::shared_ptr<isaac_ros_pointcloud_interfaces::msg::FlatScan> prepare(
    isaac_ros_pointcloud_interfaces::msg::FlatScan source,
    cudaStream_t,
    const rclcpp::Logger &)
  {
    return std::make_shared<isaac_ros_pointcloud_interfaces::msg::FlatScan>(
      std::move(source));
  }

  static std_msgs::msg::Header & header(
    isaac_ros_pointcloud_interfaces::msg::FlatScan & message)
  {
    return message.header;
  }

  static const std_msgs::msg::Header & header(
    const isaac_ros_pointcloud_interfaces::msg::FlatScan & message)
  {
    return message.header;
  }
};
ISAAC_ROS_DECLARE_BUFFER_TOPIC_ADAPTERS(
  FlatScan, isaac_ros_pointcloud_interfaces::msg::FlatScan);

}  // namespace isaac_ros_benchmark

ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(Image)
ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(CompressedImage)
ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(PointCloud2)
ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(DisparityImage)
ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(TensorList)
ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(FlatScan)
