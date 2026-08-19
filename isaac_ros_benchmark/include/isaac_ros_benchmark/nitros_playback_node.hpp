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

#ifndef ISAAC_ROS_BENCHMARK__NITROS_PLAYBACK_NODE_HPP_
#define ISAAC_ROS_BENCHMARK__NITROS_PLAYBACK_NODE_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp"

#include "ros2_benchmark/playback_node.hpp"

#include "std_msgs/msg/header.hpp"

#include "isaac_ros_benchmark/nitros_topic_adapter.hpp"

namespace isaac_ros_benchmark
{

enum class NitrosPlaybackNodePubSubType : uint8_t
{
  kNitrosType = 0,
  kGenericType
};

class NitrosPlaybackNode : public ros2_benchmark::PlaybackNode
{
public:
  /// Construct a new NitrosPlaybackNode object.
  explicit NitrosPlaybackNode(const rclcpp::NodeOptions &);

private:
  /// Create a pair of publisher and subscriber for the given data format.
  void CreateGenericPubSub(const std::string data_format, const size_t index);

  /// Create a pair of NITROS publisher and subscriber for the given data format.
  void CreateNitrosPubSub(const std::string data_format, const size_t index);

  /// Check if all the expected number of messages are buffered.
  bool AreBuffersFull() const override;

  /// Clear all the message buffers.
  void ClearBuffers() override;

  /// Publish a buffered message from the selected publisher with revised timestamps.
  bool PublishMessage(
    const size_t pub_index,
    const size_t message_index,
    const std::optional<std_msgs::msg::Header> & header) override;

  /// Get the count of all the recorded messages.
  uint64_t GetRecordedMessageCount() const override;

  /// Get the count of the recorded messages for the specified pub/sub index.
  uint64_t GetRecordedMessageCount(size_t pub_index) const override;

  /// Get the number of publishers created in this node.
  size_t GetPublisherCount() const override;

  /// A map between the pub/sub indices and their types (generic or NITROS).
  std::unordered_map<size_t, NitrosPlaybackNodePubSubType> data_format_pub_sub_types_;

  /// NITROS-type pub/sub + buffer adapters keyed by data_formats_ index. Each
  /// adapter owns a typed Publisher<T>/Subscription<T>/buffer<T> trio.
  std::unordered_map<size_t, std::unique_ptr<NitrosPlaybackTopicAdapter>> nitros_adapters_;
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__NITROS_PLAYBACK_NODE_HPP_
