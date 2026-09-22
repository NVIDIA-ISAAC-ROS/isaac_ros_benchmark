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

#ifndef ISAAC_ROS_BENCHMARK__BUFFER_MONITOR_NODE_HPP_
#define ISAAC_ROS_BENCHMARK__BUFFER_MONITOR_NODE_HPP_

#include <memory>

#include "isaac_ros_benchmark/rosidl_buffer_topic_adapter.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros2_benchmark/monitor_node.hpp"
#include "std_msgs/msg/header.hpp"

namespace isaac_ros_benchmark
{

class BufferMonitorNode : public ros2_benchmark::MonitorNode
{
public:
  explicit BufferMonitorNode(const rclcpp::NodeOptions & options);

private:
  void RecordHeader(const std_msgs::msg::Header & header);

  pluginlib::ClassLoader<BufferMonitorTopicAdapter> adapter_loader_;
  std::shared_ptr<BufferMonitorTopicAdapter> adapter_;
  std::shared_ptr<rclcpp::SubscriptionBase> subscription_;
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__BUFFER_MONITOR_NODE_HPP_
