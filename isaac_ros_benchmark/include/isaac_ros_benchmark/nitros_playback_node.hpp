// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"

#include "ros2_benchmark/playback_node.hpp"

#include "ros2_benchmark_interfaces/srv/play_messages.hpp"
#include "ros2_benchmark_interfaces/srv/start_recording.hpp"
#include "ros2_benchmark_interfaces/srv/stop_recording.hpp"

#include "isaac_ros_nitros/types/nitros_type_base.hpp"
#include "isaac_ros_nitros/types/nitros_type_manager.hpp"
#include "isaac_ros_nitros/nitros_publisher.hpp"

#include "negotiated/negotiated_publisher.hpp"


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
  /// Construct a new NitrosPlaybackNode object;
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

  /// A NITROS subscriber callback function for recording the received messages.
  void NitrosTypeRecordingSubscriberCallback(
    nvidia::isaac_ros::nitros::NitrosTypeBase & msg_base,
    std::string data_format_name,
    size_t buffer_index);

  /// Get the count of all the recorded messages.
  uint64_t GetRecordedMessageCount() const override;

  /// Get the count of the recorded messages for the specified pub/sub index.
  uint64_t GetRecordedMessageCount(size_t pub_index) const override;

  /// Get the number of publishers created in this node.
  size_t GetPublisherCount() const override;

  /// A map between the pub/sub indices and their types (generic or NITROS).
  std::unordered_map<size_t, NitrosPlaybackNodePubSubType> data_format_pub_sub_types_;

  /// Buffers for storing NITROS-typed data.
  std::unordered_map<size_t, std::vector<nvidia::isaac_ros::nitros::NitrosTypeBase>>
  nitros_msg_buffers_{};

  /// NITROS publishers that publish the buffered NITROS-typed messages (converted
  /// from the input subscribers.)
  std::unordered_map<size_t, std::shared_ptr<nvidia::isaac_ros::nitros::NitrosPublisher>>
  nitros_pubs_;

  /// NITROS type manager
  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosTypeManager> nitros_type_manager_;
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__NITROS_PLAYBACK_NODE_HPP_
