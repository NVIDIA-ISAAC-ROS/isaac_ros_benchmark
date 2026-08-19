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

#ifndef ISAAC_ROS_BENCHMARK__NITROS_MONITOR_NODE_HPP_
#define ISAAC_ROS_BENCHMARK__NITROS_MONITOR_NODE_HPP_

#include <memory>
#include <string>

#include "ros2_benchmark/monitor_node.hpp"

#include "rclcpp/rclcpp.hpp"

namespace isaac_ros_benchmark
{
class NitrosMonitorNode : public ros2_benchmark::MonitorNode
{
public:
  /// Construct a new NitrosMonitorNode object.
  explicit NitrosMonitorNode(const rclcpp::NodeOptions &);

private:
  /// Top level function for creating a monitor subscriber.
  void CreateMonitorSubscriber();

  /// Create a ROS type monitor subscriber.
  template<typename T>
  void CreateROSTypeMonitorSubscriber();

  /// Subscriber callback function for the NITROS type message monitor (that adds
  /// end timestamps).
  void OnNitrosTimestamp(uint32_t timestamp_sec, uint32_t timestamp_nsec);

  /// Subscriber callback function for the ROS type message monitor (that adds
  /// end timestamps.)
  template<typename T>
  void ROSTypeMonitorSubscriberCallback(const std::shared_ptr<T> msg);

  /// The monitor subscriber should subscribe to a NITROS type or a ROS message
  /// type. If true, subscribes via the rclcpp TypeAdapter (zero-copy with the
  /// underlying NITROS C++ type). If false (or the data format is unknown), falls
  /// back to the ROS type subscription path or the generic serialized path.
  bool use_nitros_type_monitor_sub_{true};
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__NITROS_MONITOR_NODE_HPP_
