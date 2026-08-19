// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_benchmark/nitros_monitor_node.hpp"

#include <chrono>
#include <functional>

#include "isaac_ros_benchmark/nitros_topic_adapter.hpp"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "stereo_msgs/msg/disparity_image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace isaac_ros_benchmark
{

NitrosMonitorNode::NitrosMonitorNode(const rclcpp::NodeOptions & options)
: ros2_benchmark::MonitorNode("NitrosMonitorNode", options),
  use_nitros_type_monitor_sub_(declare_parameter<bool>("use_nitros_type_monitor_sub", true))
{
  RCLCPP_INFO(
    get_logger(),
    "[NitrosMonitorNode] Starting a NITROS monitor node with a service name \"%s\"",
    monitor_service_name_.c_str());

  // Create a monitor subscriber
  CreateMonitorSubscriber();
}

void NitrosMonitorNode::CreateMonitorSubscriber()
{
  // Create a monitor subscriber
  if (use_nitros_type_monitor_sub_) {
    auto sub = CreateNitrosMonitorSubscription(
      *this,
      monitor_data_format_,
      "output",
      ros2_benchmark::kQoS,
      std::bind(
        &NitrosMonitorNode::OnNitrosTimestamp,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    if (sub) {
      monitor_sub_ = sub;
      RCLCPP_INFO(
        get_logger(),
        "[NitrosMonitorNode] Created a NITROS type monitor subscriber: format=\"%s\"",
        monitor_data_format_.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "[NitrosMonitorNode] Unknown NITROS data format \"%s\"; falling back to "
      "ROS type subscriber",
      monitor_data_format_.c_str());
  }

  #define CREATE_ROS_TYPE_MONITOR_HELPER(FORMAT_NAME, ROS_TYPE_NAME) \
    if (monitor_data_format_ == FORMAT_NAME) { \
      CreateROSTypeMonitorSubscriber<ROS_TYPE_NAME>(); \
      return; \
    }

  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_camera_info", sensor_msgs::msg::CameraInfo)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_disparity_image_32FC1", stereo_msgs::msg::DisparityImage)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_disparity_image_bgr8", stereo_msgs::msg::DisparityImage)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_rgb8", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_rgba8", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_rgb16", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_bgr8", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_bgra8", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_bgr16", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_mono8", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_mono16", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_nv12", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_nv24", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_32FC1", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_32FC3", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_image_32FC4", sensor_msgs::msg::Image)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_point_cloud", sensor_msgs::msg::PointCloud2)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_compressed_image", sensor_msgs::msg::CompressedImage)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nchw", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nhwc", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nchw_rgb_f32", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nhwc_rgb_f32", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nchw_bgr_f32", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER(
    "nitros_tensor_list_nhwc_bgr_f32", isaac_ros_tensor_list_interfaces::msg::TensorList)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_detection2_d_array", vision_msgs::msg::Detection2DArray)
  CREATE_ROS_TYPE_MONITOR_HELPER("nitros_detection3_d_array", vision_msgs::msg::Detection3DArray)

  #undef CREATE_ROS_TYPE_MONITOR_HELPER

  CreateGenericTypeMonitorSubscriber();
}

template<typename ROSMessageType>
void NitrosMonitorNode::CreateROSTypeMonitorSubscriber()
{
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  std::function<void(std::shared_ptr<ROSMessageType>)> monitor_subscriber_callback =
    std::bind(
    &NitrosMonitorNode::ROSTypeMonitorSubscriberCallback<ROSMessageType>,
    this,
    std::placeholders::_1);

  monitor_sub_ = create_subscription<ROSMessageType>(
    "output",
    ros2_benchmark::kQoS,
    monitor_subscriber_callback,
    sub_options);

  RCLCPP_INFO(
    get_logger(),
    "[NitrosMonitorNode] Created a ROS type monitor subscriber: topic=\"%s\"",
    monitor_sub_->get_topic_name());
}

template<typename T>
void NitrosMonitorNode::ROSTypeMonitorSubscriberCallback(const std::shared_ptr<T> msg)
{
  std::lock_guard<std::mutex> lock(is_monitoring_mutex_);
  if (!is_monitoring_) {
    return;
  }

  uint32_t timestamp_key;
  if (revise_timestamps_as_message_ids_) {
    timestamp_key = msg->header.stamp.sec;
  } else {
    // Use increamental numbers as timestamp keys
    timestamp_key = end_timestamps_.size();
  }
  if (record_start_timestamps_) {
    std::chrono::time_point<std::chrono::system_clock> start_timestamp(
      std::chrono::seconds(msg->header.stamp.sec) +
      std::chrono::nanoseconds(msg->header.stamp.nanosec));
    RecordStartTimestamp(timestamp_key, start_timestamp);
  }
  RecordEndTimestamp(timestamp_key);
}

void NitrosMonitorNode::OnNitrosTimestamp(uint32_t timestamp_sec, uint32_t timestamp_nsec)
{
  std::lock_guard<std::mutex> lock(is_monitoring_mutex_);
  if (!is_monitoring_) {
    return;
  }

  const uint32_t timestamp_key = revise_timestamps_as_message_ids_ ?
    timestamp_sec : static_cast<uint32_t>(end_timestamps_.size());

  if (record_start_timestamps_) {
    std::chrono::time_point<std::chrono::system_clock> start_timestamp(
      std::chrono::seconds(timestamp_sec) +
      std::chrono::nanoseconds(timestamp_nsec));
    RecordStartTimestamp(timestamp_key, start_timestamp);
  }
  RecordEndTimestamp(timestamp_key);
}

}  // namespace isaac_ros_benchmark

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::NitrosMonitorNode)
