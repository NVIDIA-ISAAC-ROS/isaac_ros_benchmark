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

#include "isaac_ros_benchmark/buffer_playback_node.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "ros2_benchmark/common.hpp"

namespace isaac_ros_benchmark
{

BufferPlaybackNode::BufferPlaybackNode(const rclcpp::NodeOptions & options)
: ros2_benchmark::PlaybackNode("BufferPlaybackNode", options),
  adapter_loader_("isaac_ros_benchmark", "isaac_ros_benchmark::BufferPlaybackTopicAdapter")
{
  if (data_formats_.empty()) {
    throw std::invalid_argument("BufferPlaybackNode requires at least one data format");
  }
  requested_buffer_length_ = 0;
  stop_recording_ = true;
  record_data_timeline_ = false;

  for (size_t index = 0; index < data_formats_.size(); ++index) {
    auto adapter = CreateAdapter(data_formats_[index]);
    if (!adapter) {
      pub_sub_types_[index] = BufferPlaybackPubSubType::kGeneric;
      ros2_benchmark::PlaybackNode::CreateGenericPubSub(data_formats_[index], index);
      continue;
    }

    pub_sub_types_[index] = BufferPlaybackPubSubType::kAdapter;
    adapter->setup(
      *this,
      "input" + std::to_string(index),
      "buffer/input" + std::to_string(index),
      ros2_benchmark::kQoS,
      ros2_benchmark::kBufferQoS,
      max_size_,
      requested_buffer_length_,
      [this, index](size_t message_index) {
        if (record_data_timeline_) {
          AddToTimestampsToMessagesMap(
            get_clock()->now().nanoseconds(), index, message_index);
        }
      });
    subs_[index] = adapter->subscription();
    adapters_[index] = std::move(adapter);
  }
}

std::shared_ptr<BufferPlaybackTopicAdapter> BufferPlaybackNode::CreateAdapter(
  const std::string & type_name)
{
  if (!adapter_loader_.isClassAvailable(type_name)) {
    return nullptr;
  }
  return adapter_loader_.createSharedInstance(type_name);
}

bool BufferPlaybackNode::AreBuffersFull() const
{
  if (record_data_timeline_) {
    return false;
  }
  if (requested_buffer_length_ == 0) {
    return timestamps_to_messages_map_.size() > 0 &&
           GetRecordedMessageCount() == GetTimestampsToMessagesCount();
  }
  for (const auto & [index, type] : pub_sub_types_) {
    if (type == BufferPlaybackPubSubType::kAdapter) {
      if (adapters_.at(index)->buffer_size() < requested_buffer_length_) {
        return false;
      }
    } else if (serialized_msg_buffers_.at(index).size() < requested_buffer_length_) {
      return false;
    }
  }
  return true;
}

void BufferPlaybackNode::ClearBuffers()
{
  for (const auto & [index, type] : pub_sub_types_) {
    if (type == BufferPlaybackPubSubType::kAdapter) {
      adapters_[index]->clear_buffer();
    } else {
      serialized_msg_buffers_[index].clear();
    }
  }
  timestamps_to_messages_map_.clear();
}

bool BufferPlaybackNode::PublishMessage(
  size_t pub_index,
  size_t message_index,
  const std::optional<std_msgs::msg::Header> & header)
{
  const auto type = pub_sub_types_.find(pub_index);
  if (type == pub_sub_types_.end()) {
    return false;
  }
  if (type->second == BufferPlaybackPubSubType::kGeneric) {
    return ros2_benchmark::PlaybackNode::PublishMessage(pub_index, message_index, header);
  }
  return adapters_.at(pub_index)->publish_at(message_index, header);
}

uint64_t BufferPlaybackNode::GetRecordedMessageCount() const
{
  uint64_t count = 0;
  for (const auto & [_, buffer] : serialized_msg_buffers_) {
    count += buffer.size();
  }
  for (const auto & [_, adapter] : adapters_) {
    count += adapter->buffer_size();
  }
  return count;
}

uint64_t BufferPlaybackNode::GetRecordedMessageCount(size_t pub_index) const
{
  const auto type = pub_sub_types_.find(pub_index);
  if (type == pub_sub_types_.end()) {
    return 0;
  }
  if (type->second == BufferPlaybackPubSubType::kAdapter) {
    return adapters_.at(pub_index)->buffer_size();
  }
  return serialized_msg_buffers_.at(pub_index).size();
}

size_t BufferPlaybackNode::GetPublisherCount() const
{
  return adapters_.size() + generic_pubs_.size();
}

}  // namespace isaac_ros_benchmark

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::BufferPlaybackNode)
