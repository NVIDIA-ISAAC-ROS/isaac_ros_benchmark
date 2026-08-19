// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>
#include <string>
#include <utility>

#include "ros2_benchmark/common.hpp"

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

  for (size_t index = 0; index < data_formats_.size(); index++) {
    const std::string & data_format = data_formats_[index];
    auto adapter = CreateNitrosPlaybackTopicAdapter(data_format);
    if (adapter == nullptr) {
      CreateGenericPubSub(data_format, index);
    } else {
      nitros_adapters_[index] = std::move(adapter);
      CreateNitrosPubSub(data_format, index);
    }
  }
}

void NitrosPlaybackNode::CreateGenericPubSub(
  const std::string data_format, const size_t index)
{
  const std::string ros_message_type = ResolveGenericRosMessageType(data_format);
  if (ros_message_type.empty()) {
    throw std::invalid_argument(
            "[NitrosPlaybackNode] Unknown data_format \"" + data_format + "\". "
            "Use a NITROS format registered in nitros_topic_adapter or a ROS type "
            "string such as sensor_msgs/msg/Image.");
  }

  data_format_pub_sub_types_[index] = NitrosPlaybackNodePubSubType::kGenericType;
  ros2_benchmark::PlaybackNode::CreateGenericPubSub(ros_message_type, index);
  if (ros_message_type != data_format) {
    RCLCPP_INFO(
      get_logger(),
      "[NitrosPlaybackNode] Created generic pub/sub for data_format=\"%s\" "
      "(ROS type \"%s\")",
      data_format.c_str(), ros_message_type.c_str());
  }
}

void NitrosPlaybackNode::CreateNitrosPubSub(
  const std::string data_format, const size_t index)
{
  data_format_pub_sub_types_[index] = NitrosPlaybackNodePubSubType::kNitrosType;

  const std::string pub_topic = "input" + std::to_string(index);
  const std::string sub_topic = "buffer/input" + std::to_string(index);

  nitros_adapters_[index]->setup(
    *this,
    pub_topic,
    sub_topic,
    ros2_benchmark::kQoS,
    ros2_benchmark::kBufferQoS,
    max_size_,
    requested_buffer_length_,
    [this, index](size_t msg_index) {
      if (record_data_timeline_) {
        const int64_t now_ns = this->get_clock()->now().nanoseconds();
        AddToTimestampsToMessagesMap(now_ns, index, msg_index);
      }
    });

  RCLCPP_INFO(
    get_logger(),
    "[NitrosPlaybackNode] Created NITROS pub/sub for data_format=\"%s\": "
    "pub topic=\"%s\", sub topic=\"%s\"",
    data_format.c_str(), pub_topic.c_str(), sub_topic.c_str());
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
  for (const auto & [index, type] : data_format_pub_sub_types_) {
    switch (type) {
      case NitrosPlaybackNodePubSubType::kGenericType:
        if (serialized_msg_buffers_.at(index).size() < requested_buffer_length_) {
          return false;
        }
        break;
      case NitrosPlaybackNodePubSubType::kNitrosType:
        if (nitros_adapters_.at(index)->buffer_size() < requested_buffer_length_) {
          return false;
        }
        break;
    }
  }
  return true;
}

void NitrosPlaybackNode::ClearBuffers()
{
  for (const auto & [index, type] : data_format_pub_sub_types_) {
    switch (type) {
      case NitrosPlaybackNodePubSubType::kGenericType:
        serialized_msg_buffers_[index].clear();
        break;
      case NitrosPlaybackNodePubSubType::kNitrosType:
        nitros_adapters_[index]->clear_buffer();
        break;
    }
  }
  timestamps_to_messages_map_.clear();
}

uint64_t NitrosPlaybackNode::GetRecordedMessageCount() const
{
  uint64_t message_count = 0;
  for (const auto & [_, buffer] : serialized_msg_buffers_) {
    message_count += buffer.size();
  }
  for (const auto & [_, adapter] : nitros_adapters_) {
    message_count += adapter->buffer_size();
  }
  return message_count;
}

uint64_t NitrosPlaybackNode::GetRecordedMessageCount(size_t pub_index) const
{
  const auto type_it = data_format_pub_sub_types_.find(pub_index);
  if (type_it == data_format_pub_sub_types_.end()) {
    return 0;
  }
  switch (type_it->second) {
    case NitrosPlaybackNodePubSubType::kGenericType:
      return serialized_msg_buffers_.at(pub_index).size();
    case NitrosPlaybackNodePubSubType::kNitrosType:
      return nitros_adapters_.at(pub_index)->buffer_size();
  }
  return 0;
}

bool NitrosPlaybackNode::PublishMessage(
  const size_t pub_index,
  const size_t message_index,
  const std::optional<std_msgs::msg::Header> & header = std::nullopt)
{
  const auto type_it = data_format_pub_sub_types_.find(pub_index);
  if (type_it == data_format_pub_sub_types_.end()) {
    return false;
  }
  switch (type_it->second) {
    case NitrosPlaybackNodePubSubType::kGenericType:
      return ros2_benchmark::PlaybackNode::PublishMessage(pub_index, message_index, header);
    case NitrosPlaybackNodePubSubType::kNitrosType: {
        const bool ok = nitros_adapters_.at(pub_index)->publish_at(message_index, header);
        if (!ok) {
          RCLCPP_ERROR(
            get_logger(),
            "[NitrosPlaybackNode] Failed to publish message index %ld for publisher index %ld. "
            "Total recorded messages = %ld",
            message_index, pub_index,
            nitros_adapters_.at(pub_index)->buffer_size());
        }
        return ok;
      }
  }
  return false;
}

size_t NitrosPlaybackNode::GetPublisherCount() const
{
  return generic_pubs_.size() + nitros_adapters_.size();
}

}  // namespace isaac_ros_benchmark

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::NitrosPlaybackNode)
