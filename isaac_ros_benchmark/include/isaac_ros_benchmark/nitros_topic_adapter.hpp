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

// Type-erases per-format Publisher<T>/Subscription<T>/buffer<T> so one
// NitrosPlaybackNode can multiplex pub/sub triples of multiple NITROS C++ types.

#ifndef ISAAC_ROS_BENCHMARK__NITROS_TOPIC_ADAPTER_HPP_
#define ISAAC_ROS_BENCHMARK__NITROS_TOPIC_ADAPTER_HPP_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"

namespace isaac_ros_benchmark
{

class NitrosPlaybackTopicAdapter
{
public:
  using RecordCallback = std::function<void(size_t)>;

  virtual ~NitrosPlaybackTopicAdapter() = default;

  /// Setup the publisher and subscriber.
  virtual void setup(
    rclcpp::Node & node,
    const std::string & pub_topic,
    const std::string & sub_topic,
    const rclcpp::QoS & pub_qos,
    const rclcpp::QoS & sub_qos,
    const size_t & max_buffer_size,
    const size_t & requested_buffer_length,
    RecordCallback on_record) = 0;

  virtual bool publish_at(
    size_t msg_idx,
    const std::optional<std_msgs::msg::Header> & header) = 0;

  virtual size_t buffer_size() const = 0;
  virtual void clear_buffer() = 0;
};

template<typename NitrosT>
class TypedNitrosPlaybackTopicAdapter : public NitrosPlaybackTopicAdapter
{
public:
  void setup(
    rclcpp::Node & node,
    const std::string & pub_topic,
    const std::string & sub_topic,
    const rclcpp::QoS & pub_qos,
    const rclcpp::QoS & sub_qos,
    const size_t & max_buffer_size,
    const size_t & requested_buffer_length,
    RecordCallback on_record) override
  {
    max_buffer_size_ = &max_buffer_size;
    requested_buffer_length_ = &requested_buffer_length;

    rclcpp::PublisherOptions pub_options;
    pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    pub_ = node.create_publisher<NitrosT>(pub_topic, pub_qos, pub_options);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

    sub_ = node.create_subscription<NitrosT>(
      sub_topic, sub_qos,
      [this, on_record = std::move(on_record)](std::shared_ptr<const NitrosT> msg) {
        if (*max_buffer_size_ > 0 && buffer_.size() >= *max_buffer_size_) {
          return;
        }
        if (*requested_buffer_length_ > 0 && buffer_.size() >= *requested_buffer_length_) {
          return;
        }
        buffer_.push_back(msg);
        on_record(buffer_.size() - 1);
      },
      sub_options);
  }

  bool publish_at(
    size_t msg_idx,
    const std::optional<std_msgs::msg::Header> & header) override
  {
    if (msg_idx >= buffer_.size() || !pub_) {
      return false;
    }
    NitrosT msg_copy = *buffer_[msg_idx];
    if (header) {
      msg_copy.set_timestamp_sec(header->stamp.sec);
      msg_copy.set_timestamp_nsec(header->stamp.nanosec);
    }
    pub_->publish(std::move(msg_copy));
    return true;
  }

  size_t buffer_size() const override {return buffer_.size();}
  void clear_buffer() override {buffer_.clear();}

private:
  const size_t * max_buffer_size_{nullptr};
  const size_t * requested_buffer_length_{nullptr};
  typename rclcpp::Publisher<NitrosT>::SharedPtr pub_;
  typename rclcpp::Subscription<NitrosT>::SharedPtr sub_;
  std::vector<std::shared_ptr<const NitrosT>> buffer_;
};

/// Map a NITROS data-format label to a ROS generic pub/sub type string
/// (package/msg/Type). Pass-through when \p data_format already contains '/'.
/// Returns empty string when unknown.
std::string ResolveGenericRosMessageType(const std::string & data_format);

/// Create a playback adapter for a known NITROS data format string.
/// Returns nullptr when the format is unknown (caller falls back to the generic
/// serialized PlaybackNode path).
std::unique_ptr<NitrosPlaybackTopicAdapter> CreateNitrosPlaybackTopicAdapter(
  const std::string & data_format);

/// Create a NITROS subscriber that forwards each message's (sec, nsec) timestamp.
/// Returns nullptr when the format is unknown (caller falls back to the generic
/// ROS subscription path).
std::shared_ptr<rclcpp::SubscriptionBase> CreateNitrosMonitorSubscription(
  rclcpp::Node & node,
  const std::string & data_format,
  const std::string & topic,
  const rclcpp::QoS & qos,
  std::function<void(uint32_t /*sec*/, uint32_t /*nsec*/)> on_message);

}  // namespace isaac_ros_benchmark

#endif  // ISAAC_ROS_BENCHMARK__NITROS_TOPIC_ADAPTER_HPP_
