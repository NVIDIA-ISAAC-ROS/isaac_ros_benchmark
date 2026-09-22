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

#include "isaac_ros_benchmark/buffer_monitor_node.hpp"

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>

#include "ros2_benchmark/common.hpp"

namespace isaac_ros_benchmark
{

BufferMonitorNode::BufferMonitorNode(const rclcpp::NodeOptions & options)
: ros2_benchmark::MonitorNode("BufferMonitorNode", options),
  adapter_loader_("isaac_ros_benchmark", "isaac_ros_benchmark::BufferMonitorTopicAdapter")
{
  if (!adapter_loader_.isClassAvailable(monitor_data_format_)) {
    throw std::invalid_argument(
            "No buffer message adapter registered for " + monitor_data_format_);
  }
  adapter_ = adapter_loader_.createSharedInstance(monitor_data_format_);

  subscription_ = adapter_->create_subscription(
    *this, "output", ros2_benchmark::kQoS,
    [this](const std_msgs::msg::Header & header) {RecordHeader(header);});
}

void BufferMonitorNode::RecordHeader(const std_msgs::msg::Header & header)
{
  std::lock_guard<std::mutex> lock(is_monitoring_mutex_);
  if (!is_monitoring_) {
    return;
  }

  uint32_t timestamp_key = revise_timestamps_as_message_ids_ ?
    header.stamp.sec : static_cast<uint32_t>(end_timestamps_.size());
  RecordEndTimestamp(timestamp_key);
  if (record_start_timestamps_) {
    std::chrono::time_point<std::chrono::system_clock> start_timestamp(
      std::chrono::seconds(header.stamp.sec) +
      std::chrono::nanoseconds(header.stamp.nanosec));
    RecordStartTimestamp(timestamp_key, start_timestamp);
  }
}

}  // namespace isaac_ros_benchmark

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::BufferMonitorNode)
