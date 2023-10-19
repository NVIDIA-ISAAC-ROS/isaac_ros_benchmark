// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_benchmark/nitros_playback_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "gxf/core/gxf.h"
#include "extensions/gxf_optimizer/exporter/graph_types.hpp"

#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

#include "isaac_ros_nitros_april_tag_detection_array_type/nitros_april_tag_detection_array.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"
#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_flat_scan_type/nitros_flat_scan.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_occupancy_grid_type/nitros_occupancy_grid.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "ros2_benchmark/common.hpp"
#include "ros2_benchmark_interfaces/msg/timestamped_message_array.hpp"
#include "ros2_benchmark_interfaces/msg/topic_message_count.hpp"

namespace isaac_ros_benchmark
{

NitrosPlaybackNode::NitrosPlaybackNode(const rclcpp::NodeOptions & options)
: ros2_benchmark::PlaybackNode("NitrosPlaybackNode", options)
{
  if (data_formats_.empty()) {
    throw std::invalid_argument(
            "[NitrosPlaybackNode] Empty data_formats, "
            "this needs to be set to match the data formats for the data to be buffered");
  }

  // Create a Nitros type manager for the node
  nitros_type_manager_ = std::make_shared<nvidia::isaac_ros::nitros::NitrosTypeManager>(this);

  nitros_type_manager_->registerSupportedType<
    nvidia::isaac_ros::nitros::NitrosAprilTagDetectionArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection2DArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosFlatScan>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosPointCloud>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosCompressedImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosOccupancyGrid>();

  for (size_t index = 0; index < data_formats_.size(); index++) {
    std::string data_format = data_formats_[index];

    if (!nitros_type_manager_->hasFormat(data_format)) {
      CreateGenericPubSub(data_format, index);
    } else {
      CreateNitrosPubSub(data_format, index);
    }
  }
}

void NitrosPlaybackNode::CreateGenericPubSub(
  const std::string data_format, const size_t index)
{
  data_format_pub_sub_types_[index] = NitrosPlaybackNodePubSubType::kGenericType;
  ros2_benchmark::PlaybackNode::CreateGenericPubSub(data_format, index);
}

void NitrosPlaybackNode::CreateNitrosPubSub(
  const std::string data_format, const size_t index)
{
  // Load the extensions needed by the used NITROS format
  RCLCPP_INFO(
    get_logger(),
    "Loading extensions for NITROS format %s",
    data_format.c_str());
  nitros_type_manager_->loadExtensions(data_format);

  data_format_pub_sub_types_[index] = NitrosPlaybackNodePubSubType::kNitrosType;

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  nvidia::gxf::optimizer::ComponentInfo empty_component_info = {"", "", ""};

  // Create a Nitros (negotiated) publisher
  negotiated::NegotiatedPublisherOptions negotiated_pub_options;
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wpedantic"
  nvidia::isaac_ros::nitros::NitrosPublisherSubscriberConfig config = {
    .type = nvidia::isaac_ros::nitros::NitrosPublisherSubscriberType::NEGOTIATED,
    .qos = ros2_benchmark::kQoS,
    .compatible_data_format = data_format,
    .topic_name = "input" + std::to_string(index),
    // .callback = nullptr
  };
  #pragma GCC diagnostic pop

  auto nitros_pub = std::make_shared<nvidia::isaac_ros::nitros::NitrosPublisher>(
    *this,
    nitros_type_manager_,
    empty_component_info,
    std::vector<std::string>{data_format},
    config,
    negotiated_pub_options);
  nitros_pub->setContext(
    nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getContext());
  nitros_pub->start();
  nitros_pubs_[index] = nitros_pub;

  RCLCPP_INFO(
    get_logger(),
    "[NitrosPlaybackNode] Created a NITROS publisher: topic=\"%s\"",
    config.topic_name.c_str());
  RCLCPP_INFO(get_logger(), "[NitrosPlaybackNode] Negotiation started...");

  // Create a recording subscriber
  if (!nitros_type_manager_->hasFormat(data_format)) {
    std::stringstream error_msg;
    error_msg <<
      "[NitrosPlaybackNode] Could not identify the recording subscriber data foramt: " <<
      "\"" << data_format.c_str() << "\"";
    RCLCPP_ERROR(
      get_logger(), error_msg.str().c_str());
    throw std::runtime_error(error_msg.str().c_str());
  }

  std::function<void(nvidia::isaac_ros::nitros::NitrosTypeBase &, const std::string)>
  subscriber_callback =
    std::bind(
    &NitrosPlaybackNode::NitrosTypeRecordingSubscriberCallback,
    this,
    std::placeholders::_1,
    std::placeholders::_2,
    index);

  std::shared_ptr<rclcpp::SubscriptionBase> sub;
  nitros_type_manager_->getFormatCallbacks(data_format)
  .createCompatibleSubscriberCallback(
    *this,
    sub,
    "buffer/input" + std::to_string(index),
    ros2_benchmark::kBufferQoS,
    subscriber_callback,
    sub_options
  );
  subs_[index] = sub;
  RCLCPP_INFO(
    get_logger(), "[NitrosPlaybackNode] Created a NITROS subscriber: topic=\"%s\"",
    sub->get_topic_name());

  // Create a NITROS message buffer
  nitros_msg_buffers_[index] = std::vector<nvidia::isaac_ros::nitros::NitrosTypeBase>();
}

bool NitrosPlaybackNode::AreBuffersFull() const
{
  if (record_data_timeline_) {
    return false;
  }
  if (requested_buffer_length_ == 0) {
    if ((timestamps_to_messages_map_.size() > 0) &&
      (GetRecordedMessageCount() == GetTimestampsToMessagesCount()))
    {
      return true;
    }
    return false;
  }
  for (size_t i = 0; i < data_format_pub_sub_types_.size(); i++) {
    switch (data_format_pub_sub_types_.at(i)) {
      case NitrosPlaybackNodePubSubType::kGenericType:
        if (serialized_msg_buffers_.at(i).size() < requested_buffer_length_) {
          return false;
        }
        break;
      case NitrosPlaybackNodePubSubType::kNitrosType:
        if (nitros_msg_buffers_.at(i).size() < requested_buffer_length_) {
          return false;
        }
        break;
    }
  }
  return true;
}

void NitrosPlaybackNode::ClearBuffers()
{
  for (size_t i = 0; i < data_format_pub_sub_types_.size(); i++) {
    switch (data_format_pub_sub_types_[i]) {
      case NitrosPlaybackNodePubSubType::kGenericType:
        serialized_msg_buffers_[i].clear();
        break;
      case NitrosPlaybackNodePubSubType::kNitrosType:
        nitros_msg_buffers_[i].clear();
        break;
    }
  }
  timestamps_to_messages_map_.clear();
}

uint64_t NitrosPlaybackNode::GetRecordedMessageCount() const
{
  uint64_t message_count = 0;
  for (const auto & buffer : serialized_msg_buffers_) {
    message_count += buffer.second.size();
  }
  for (const auto & buffer : nitros_msg_buffers_) {
    message_count += buffer.second.size();
  }
  return message_count;
}

uint64_t NitrosPlaybackNode::GetRecordedMessageCount(size_t pub_index) const
{
  switch (data_format_pub_sub_types_.at(pub_index)) {
    case NitrosPlaybackNodePubSubType::kGenericType:
      return serialized_msg_buffers_.at(pub_index).size();
    case NitrosPlaybackNodePubSubType::kNitrosType:
      return nitros_msg_buffers_.at(pub_index).size();
    default:
      return 0;
  }
}

bool NitrosPlaybackNode::PublishMessage(
  const size_t pub_index,
  const size_t message_index,
  const std::optional<std_msgs::msg::Header> & header = std::nullopt)
{
  switch (data_format_pub_sub_types_[pub_index]) {
    case NitrosPlaybackNodePubSubType::kGenericType:
      return ros2_benchmark::PlaybackNode::PublishMessage(pub_index, message_index, header);

    case NitrosPlaybackNodePubSubType::kNitrosType:
      size_t buffer_size = nitros_msg_buffers_[pub_index].size();
      if (message_index >= buffer_size) {
        RCLCPP_ERROR(
          get_logger(),
          "[NitrosPlaybackNode] Failed to publish message index %ld for publishre index %ld. " \
          "Total recorded messages = %ld",
          message_index, pub_index, buffer_size);
        return false;
      }
      auto nitros_type_msg = nitros_msg_buffers_[pub_index].at(message_index);
      if (header) {
        nitros_pubs_[pub_index]->publish(nitros_type_msg, *header);
      } else {
        nitros_pubs_[pub_index]->publish(nitros_type_msg);
      }
      return true;
  }
  return false;
}

void NitrosPlaybackNode::NitrosTypeRecordingSubscriberCallback(
  nvidia::isaac_ros::nitros::NitrosTypeBase & msg_base,
  std::string data_format_name,
  size_t buffer_index)
{
  (void)data_format_name;
  if ((nitros_msg_buffers_[buffer_index].size() >= max_size_) ||
    (requested_buffer_length_ > 0 &&
    nitros_msg_buffers_[buffer_index].size() >= requested_buffer_length_))
  {
    RCLCPP_DEBUG(get_logger(), "[NitrosPlaybackNode] Dropped a message due to a full buffer");
    return;
  }
  // Add received Nitros-typed message to internal buffer
  nitros_msg_buffers_[buffer_index].push_back(msg_base);
  RCLCPP_DEBUG(
    get_logger(), "[NitrosPlaybackNode] Added a message to the buffer (%ld/%ld)",
    nitros_msg_buffers_[buffer_index].size(), requested_buffer_length_);

  if (record_data_timeline_) {
    int64_t now_ns = this->get_clock()->now().nanoseconds();
    size_t message_index = serialized_msg_buffers_[buffer_index].size() - 1;
    AddToTimestampsToMessagesMap(now_ns, buffer_index, message_index);
  }
}

size_t NitrosPlaybackNode::GetPublisherCount() const
{
  return generic_pubs_.size() + nitros_pubs_.size();
}

}  // namespace isaac_ros_benchmark

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::NitrosPlaybackNode)
