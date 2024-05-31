// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include <memory>
#include <string>

#include "ros2_benchmark/monitor_node.hpp"

#include "rclcpp/rclcpp.hpp"

#include "isaac_ros_nitros/nitros_subscriber.hpp"
#include "isaac_ros_nitros/types/nitros_type_base.hpp"
#include "isaac_ros_nitros/types/nitros_type_manager.hpp"

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

  /// Create a NITROS type monitor subscriber.
  void CreateNitrosMonitorSubscriber();

  /// Subscriber callback function for the ROS type message monitor (that adds
  /// end timestamps.)
  template<typename T>
  void ROSTypeMonitorSubscriberCallback(const std::shared_ptr<T> msg);

  /// Subscriber callback function for the Nitros type message monitor (that adds
  /// end timestamps.)
  void NitrosTypeMonitorSubscriberCallback(
    const gxf_context_t,
    nvidia::isaac_ros::nitros::NitrosTypeBase & msg_base);

  /// The monitor subscriber should subscriber to a Nitros type or a ROS message type.
  bool use_nitros_type_monitor_sub_{true};

  // Nitros subscriber for monitoring incoming NITROS type messages
  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosSubscriber> nitros_sub_;

  /// Nitros type manager.
  std::shared_ptr<nvidia::isaac_ros::nitros::NitrosTypeManager> nitros_type_manager_;
};

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__NITROS_MONITOR_NODE_HPP_
