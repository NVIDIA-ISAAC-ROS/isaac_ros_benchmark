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

#ifndef ISAAC_ROS_BENCHMARK__BUFFER_PLAYBACK_NODE_HPP_
#define ISAAC_ROS_BENCHMARK__BUFFER_PLAYBACK_NODE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "isaac_ros_benchmark/rosidl_buffer_topic_adapter.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros2_benchmark/playback_node.hpp"
#include "std_msgs/msg/header.hpp"

namespace isaac_ros_benchmark
{

enum class BufferPlaybackPubSubType : uint8_t
{
  kAdapter = 0,
  kGeneric
};

class BufferPlaybackNode : public ros2_benchmark::PlaybackNode
{
public:
  explicit BufferPlaybackNode(const rclcpp::NodeOptions & options);

private:
  std::shared_ptr<BufferPlaybackTopicAdapter> CreateAdapter(const std::string & type_name);
  bool AreBuffersFull() const override;
  void ClearBuffers() override;
  bool PublishMessage(
    size_t pub_index,
    size_t message_index,
    const std::optional<std_msgs::msg::Header> & header) override;
  uint64_t GetRecordedMessageCount() const override;
  uint64_t GetRecordedMessageCount(size_t pub_index) const override;
  size_t GetPublisherCount() const override;

  pluginlib::ClassLoader<BufferPlaybackTopicAdapter> adapter_loader_;
  std::unordered_map<size_t, BufferPlaybackPubSubType> pub_sub_types_;
  std::unordered_map<size_t, std::shared_ptr<BufferPlaybackTopicAdapter>> adapters_;
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__BUFFER_PLAYBACK_NODE_HPP_
