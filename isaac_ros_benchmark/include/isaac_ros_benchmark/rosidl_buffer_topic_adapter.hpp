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

#ifndef ISAAC_ROS_BENCHMARK__ROSIDL_BUFFER_TOPIC_ADAPTER_HPP_
#define ISAAC_ROS_BENCHMARK__ROSIDL_BUFFER_TOPIC_ADAPTER_HPP_

#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "cuda_buffer/cuda_buffer_impl.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "std_msgs/msg/header.hpp"

namespace isaac_ros_benchmark
{

class BufferPlaybackTopicAdapter
{
public:
  using RecordCallback = std::function<void(size_t)>;

  virtual ~BufferPlaybackTopicAdapter() = default;

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
    size_t message_index,
    const std::optional<std_msgs::msg::Header> & header) = 0;

  virtual std::shared_ptr<rclcpp::SubscriptionBase> subscription() const = 0;
  virtual size_t buffer_size() const = 0;
  virtual void clear_buffer() = 0;
};

class BufferMonitorTopicAdapter
{
public:
  using HeaderCallback = std::function<void(const std_msgs::msg::Header &)>;

  virtual ~BufferMonitorTopicAdapter() = default;

  virtual std::shared_ptr<rclcpp::SubscriptionBase> create_subscription(
    rclcpp::Node & node,
    const std::string & topic,
    const rclcpp::QoS & qos,
    HeaderCallback on_message) = 0;
};

template<typename MessageT>
struct RosidlBufferMessageTraits;

template<typename T>
cudaError_t clone_buffer_to_cuda(
  rosidl::Buffer<T> & source,
  cudaStream_t stream)
{
  rosidl::Buffer<T> destination(
    std::make_unique<cuda_buffer_backend::CudaBufferImpl<T>>(source.size()));
  if (source.size() != 0) {
    auto input = cuda_buffer_backend::from_input_buffer(source, stream);
    auto output = cuda_buffer_backend::from_output_buffer(destination, stream);
    const cudaError_t error = cudaMemcpyAsync(
      output.get_ptr(), input.get_ptr(), source.size() * sizeof(T),
      cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
      return error;
    }
  }
  source = std::move(destination);
  return cudaSuccess;
}

template<typename MessageT>
std::shared_ptr<MessageT> finish_buffering(
  MessageT source,
  cudaStream_t stream,
  const rclcpp::Logger & logger,
  cudaError_t error)
{
  if (error == cudaSuccess) {
    error = cudaStreamSynchronize(stream);
  }
  if (error != cudaSuccess) {
    const std::string type_name = rosidl_generator_traits::name<MessageT>();
    RCLCPP_ERROR(
      logger, "Failed to buffer %s: %s",
      type_name.c_str(), cudaGetErrorString(error));
    return nullptr;
  }
  return std::make_shared<MessageT>(std::move(source));
}

template<typename MessageT, auto BufferMember, auto HeaderMember>
struct SingleRosidlBufferMessageTraits
{
  using BufferType = std::remove_cv_t<std::remove_reference_t<
        decltype(std::declval<MessageT &>().*BufferMember)>>;
  using HeaderType = std::remove_cv_t<std::remove_reference_t<
        decltype(std::declval<MessageT &>().*HeaderMember)>>;

  static_assert(
    std::is_same_v<BufferType, rosidl::Buffer<uint8_t>>,
    "BufferMember must refer to rosidl::Buffer<uint8_t>");
  static_assert(
    std::is_same_v<HeaderType, std_msgs::msg::Header>,
    "HeaderMember must refer to std_msgs::msg::Header");

  static constexpr bool kHasBufferBackend = true;

  static std::string type_name()
  {
    return rosidl_generator_traits::name<MessageT>();
  }

  static std::shared_ptr<MessageT> prepare(
    MessageT source,
    cudaStream_t stream,
    const rclcpp::Logger & logger)
  {
    const cudaError_t error = clone_buffer_to_cuda(buffer(source), stream);
    return finish_buffering(std::move(source), stream, logger, error);
  }

  static BufferType & buffer(MessageT & message)
  {
    return message.*BufferMember;
  }

  static const BufferType & buffer(const MessageT & message)
  {
    return message.*BufferMember;
  }

  static std_msgs::msg::Header & header(MessageT & message)
  {
    return message.*HeaderMember;
  }

  static const std_msgs::msg::Header & header(const MessageT & message)
  {
    return message.*HeaderMember;
  }
};

template<typename MessageT>
class TypedRosidlBufferPlaybackTopicAdapter : public BufferPlaybackTopicAdapter
{
public:
  TypedRosidlBufferPlaybackTopicAdapter()
  {
    if constexpr (RosidlBufferMessageTraits<MessageT>::kHasBufferBackend) {
      const cudaError_t error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
      if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }
  }

  ~TypedRosidlBufferPlaybackTopicAdapter() override
  {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

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
    publisher_ = node.create_publisher<MessageT>(pub_topic, pub_qos, pub_options);

    subscription_ = node.create_generic_subscription(
      sub_topic, RosidlBufferMessageTraits<MessageT>::type_name(), sub_qos,
      [this, logger = node.get_logger(), on_record = std::move(on_record)](
        std::shared_ptr<rclcpp::SerializedMessage> serialized)
      {
        MessageT source;
        rclcpp::Serialization<MessageT> serializer;
        serializer.deserialize_message(serialized.get(), &source);

        std::lock_guard<std::mutex> lock(mutex_);
        if ((*max_buffer_size_ > 0 && buffer_.size() >= *max_buffer_size_) ||
        (*requested_buffer_length_ > 0 &&
        buffer_.size() >= *requested_buffer_length_))
        {
          return;
        }
        std::shared_ptr<MessageT> message;
        try {
          message =
          RosidlBufferMessageTraits<MessageT>::prepare(std::move(source), stream_, logger);
        } catch (const std::exception & exception) {
          RCLCPP_ERROR(logger, "Failed to buffer message: %s", exception.what());
          return;
        }
        if (!message) {
          return;
        }
        buffer_.push_back(std::move(message));
        on_record(buffer_.size() - 1);
      });
  }

  bool publish_at(
    size_t message_index,
    const std::optional<std_msgs::msg::Header> & header) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!publisher_ || message_index >= buffer_.size()) {
      return false;
    }
    MessageT & message = *buffer_[message_index];
    std_msgs::msg::Header & message_header =
      RosidlBufferMessageTraits<MessageT>::header(message);
    const std_msgs::msg::Header original_header = message_header;
    if (header) {
      message_header = *header;
    }
    publisher_->publish(message);
    message_header = original_header;
    return true;
  }

  std::shared_ptr<rclcpp::SubscriptionBase> subscription() const override
  {
    return subscription_;
  }

  size_t buffer_size() const override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
  }

  void clear_buffer() override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
  }

private:
  cudaStream_t stream_{nullptr};
  const size_t * max_buffer_size_{nullptr};
  const size_t * requested_buffer_length_{nullptr};
  typename rclcpp::Publisher<MessageT>::SharedPtr publisher_;
  std::shared_ptr<rclcpp::SubscriptionBase> subscription_;
  std::vector<std::shared_ptr<MessageT>> buffer_;
  mutable std::mutex mutex_;
};

template<typename MessageT>
class TypedRosidlBufferMonitorTopicAdapter : public BufferMonitorTopicAdapter
{
public:
  std::shared_ptr<rclcpp::SubscriptionBase> create_subscription(
    rclcpp::Node & node,
    const std::string & topic,
    const rclcpp::QoS & qos,
    HeaderCallback on_message) override
  {
    rclcpp::SubscriptionOptions options;
    options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
    if constexpr (RosidlBufferMessageTraits<MessageT>::kHasBufferBackend) {
      options.acceptable_buffer_backends = "any";
    }
    return node.create_subscription<MessageT>(
      topic, qos,
      [on_message = std::move(on_message)](const std::shared_ptr<const MessageT> & message) {
        on_message(RosidlBufferMessageTraits<MessageT>::header(*message));
      },
      options);
  }
};

#define ISAAC_ROS_DECLARE_BUFFER_TOPIC_ADAPTERS(AdapterName, MessageT) \
  using AdapterName ## BufferPlaybackTopicAdapter = \
    TypedRosidlBufferPlaybackTopicAdapter<MessageT>; \
  using AdapterName ## BufferMonitorTopicAdapter = \
    TypedRosidlBufferMonitorTopicAdapter<MessageT>

#define ISAAC_ROS_DECLARE_SINGLE_BUFFER_TOPIC_ADAPTERS( \
    AdapterName, MessageT, BufferField, HeaderField) \
  template<> \
  struct RosidlBufferMessageTraits<MessageT> \
    : SingleRosidlBufferMessageTraits< \
      MessageT, &MessageT::BufferField, &MessageT::HeaderField> {}; \
  ISAAC_ROS_DECLARE_BUFFER_TOPIC_ADAPTERS(AdapterName, MessageT)

}  // namespace isaac_ros_benchmark

#define ISAAC_ROS_EXPORT_BUFFER_TOPIC_ADAPTERS(AdapterName) \
  PLUGINLIB_EXPORT_CLASS( \
    isaac_ros_benchmark::AdapterName ## BufferPlaybackTopicAdapter, \
    isaac_ros_benchmark::BufferPlaybackTopicAdapter) \
  PLUGINLIB_EXPORT_CLASS( \
    isaac_ros_benchmark::AdapterName ## BufferMonitorTopicAdapter, \
    isaac_ros_benchmark::BufferMonitorTopicAdapter)

#endif  // ISAAC_ROS_BENCHMARK__ROSIDL_BUFFER_TOPIC_ADAPTER_HPP_
