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

#include "isaac_ros_benchmark/nitros_topic_adapter.hpp"

#include <initializer_list>

#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_flat_scan_type/nitros_flat_scan.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace isaac_ros_benchmark
{

namespace
{

#define FOR_EACH_NITROS_TYPE_FORMAT(ENTRY) \
  ENTRY(NitrosCompressedImage, "nitros_compressed_image") \
  ENTRY(NitrosDisparityImage, "nitros_disparity_image_32FC1", "nitros_disparity_image_bgr8") \
  ENTRY(NitrosFlatScan, "nitros_flat_scan") \
  ENTRY(NitrosImage, \
    "nitros_image_rgb8", "nitros_image_rgba8", "nitros_image_rgb16", \
    "nitros_image_bgr8", "nitros_image_bgra8", "nitros_image_bgr16", \
    "nitros_image_mono8", "nitros_image_mono16", \
    "nitros_image_nv12", "nitros_image_nv24", \
    "nitros_image_32FC1", "nitros_image_32FC3", "nitros_image_32FC4") \
  ENTRY(NitrosPointCloud, "nitros_point_cloud") \
  ENTRY(NitrosTensorList, \
    "nitros_tensor_list_nchw", "nitros_tensor_list_nhwc", \
    "nitros_tensor_list_nchw_rgb_f32", "nitros_tensor_list_nhwc_rgb_f32", \
    "nitros_tensor_list_nchw_bgr_f32", "nitros_tensor_list_nhwc_bgr_f32")

bool MatchesAny(const std::string & key, std::initializer_list<const char *> options)
{
  for (const char * opt : options) {
    if (key == opt) {return true;}
  }
  return false;
}

template<typename NitrosT>
std::shared_ptr<rclcpp::SubscriptionBase> MakeTimestampSubscription(
  rclcpp::Node & node,
  const std::string & topic,
  const rclcpp::QoS & qos,
  std::function<void(uint32_t, uint32_t)> on_message)
{
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  return node.create_subscription<NitrosT>(
    topic, qos,
    [on_message = std::move(on_message)](std::shared_ptr<const NitrosT> msg) {
      on_message(msg->get_timestamp_sec(), msg->get_timestamp_nsec());
    },
    sub_options);
}

}  // namespace

std::string ResolveGenericRosMessageType(const std::string & data_format)
{
  if (data_format.find('/') != std::string::npos) {
    return data_format;
  }

#define MAP_NITROS_FORMAT_TO_ROS_MSG(FMT, ROS_MSG) \
  if (data_format == FMT) {return ROS_MSG;}

  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_camera_info", "sensor_msgs/msg/CameraInfo")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_detection2_d_array", "vision_msgs/msg/Detection2DArray")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_detection3_d_array", "vision_msgs/msg/Detection3DArray")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_disparity_image_32FC1", "stereo_msgs/msg/DisparityImage")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_disparity_image_bgr8", "stereo_msgs/msg/DisparityImage")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_point_cloud", "sensor_msgs/msg/PointCloud2")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_compressed_image", "sensor_msgs/msg/CompressedImage")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_flat_scan", "isaac_ros_pointcloud_interfaces/msg/FlatScan")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_rgb8", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_rgba8", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_rgb16", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_bgr8", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_bgra8", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_bgr16", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_mono8", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_mono16", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_nv12", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_nv24", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_32FC1", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_32FC3", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG("nitros_image_32FC4", "sensor_msgs/msg/Image")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nchw", "isaac_ros_tensor_list_interfaces/msg/TensorList")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nhwc", "isaac_ros_tensor_list_interfaces/msg/TensorList")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nchw_rgb_f32", "isaac_ros_tensor_list_interfaces/msg/TensorList")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nhwc_rgb_f32", "isaac_ros_tensor_list_interfaces/msg/TensorList")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nchw_bgr_f32", "isaac_ros_tensor_list_interfaces/msg/TensorList")
  MAP_NITROS_FORMAT_TO_ROS_MSG(
    "nitros_tensor_list_nhwc_bgr_f32", "isaac_ros_tensor_list_interfaces/msg/TensorList")

#undef MAP_NITROS_FORMAT_TO_ROS_MSG

  return {};
}

std::unique_ptr<NitrosPlaybackTopicAdapter> CreateNitrosPlaybackTopicAdapter(
  const std::string & data_format)
{
  using namespace nvidia::isaac_ros::nitros;  // NOLINT

  #define MAKE_PLAYBACK_ADAPTER(TYPE, ...) \
    if (MatchesAny(data_format, {__VA_ARGS__})) { \
      return std::make_unique<TypedNitrosPlaybackTopicAdapter<TYPE>>(); \
    }
  FOR_EACH_NITROS_TYPE_FORMAT(MAKE_PLAYBACK_ADAPTER)
  #undef MAKE_PLAYBACK_ADAPTER

  return nullptr;
}

std::shared_ptr<rclcpp::SubscriptionBase> CreateNitrosMonitorSubscription(
  rclcpp::Node & node,
  const std::string & data_format,
  const std::string & topic,
  const rclcpp::QoS & qos,
  std::function<void(uint32_t, uint32_t)> on_message)
{
  using namespace nvidia::isaac_ros::nitros;  // NOLINT

  #define MAKE_MONITOR_SUBSCRIPTION(TYPE, ...) \
    if (MatchesAny(data_format, {__VA_ARGS__})) { \
      return MakeTimestampSubscription<TYPE>(node, topic, qos, std::move(on_message)); \
    }
  FOR_EACH_NITROS_TYPE_FORMAT(MAKE_MONITOR_SUBSCRIPTION)
  #undef MAKE_MONITOR_SUBSCRIPTION

  return nullptr;
}

}  // namespace isaac_ros_benchmark
