// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_benchmark/nitros_monitor_node.hpp"

#include "isaac_ros_nitros/types/type_adapter_nitros_context.hpp"

#include "isaac_ros_nitros_april_tag_detection_array_type/nitros_april_tag_detection_array.hpp"
#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_detection2_d_array_type/nitros_detection2_d_array.hpp"
#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_point_cloud_type/nitros_point_cloud.hpp"
#include "isaac_ros_nitros_pose_cov_stamped_type/nitros_pose_cov_stamped.hpp"
#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "isaac_ros_nitros_occupancy_grid_type/nitros_occupancy_grid.hpp"

namespace isaac_ros_benchmark
{

NitrosMonitorNode::NitrosMonitorNode(const rclcpp::NodeOptions & options)
: ros2_benchmark::MonitorNode("NitrosMonitorNode", options),
  use_nitros_type_monitor_sub_(declare_parameter<bool>("use_nitros_type_monitor_sub", true))
{
  RCLCPP_INFO(
    get_logger(),
    "[NitrosMonitorNode] Starting a NITROS monitor node with a service name \"%s\"",
    monitor_service_name_.c_str());

  // Create a Nitros type manager for the node
  nitros_type_manager_ = std::make_shared<nvidia::isaac_ros::nitros::NitrosTypeManager>(this);

  nitros_type_manager_->registerSupportedType<
    nvidia::isaac_ros::nitros::NitrosAprilTagDetectionArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection2DArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosDisparityImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosPointCloud>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosPoseCovStamped>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosCompressedImage>();
  nitros_type_manager_->registerSupportedType<nvidia::isaac_ros::nitros::NitrosOccupancyGrid>();

  // Create a monitor subscriber
  CreateMonitorSubscriber();
}

void NitrosMonitorNode::CreateMonitorSubscriber()
{
  // Create a monitor subscriber
  if (!nitros_type_manager_->hasFormat(monitor_data_format_)) {
    CreateGenericTypeMonitorSubscriber();
    return;
  }

  if (use_nitros_type_monitor_sub_) {
    CreateNitrosMonitorSubscriber();
  } else {
    std::string ros_type_name =
      nitros_type_manager_->getFormatCallbacks(monitor_data_format_).getROSTypeName();

    #define CREATE_ROS_TYPE_MONITOR_HELPER(ROS_TYPE_NAME) \
  if (ros_type_name == rosidl_generator_traits::name<ROS_TYPE_NAME>()) { \
    CreateROSTypeMonitorSubscriber<ROS_TYPE_NAME>(); \
    return; \
  }

    CREATE_ROS_TYPE_MONITOR_HELPER(isaac_ros_apriltag_interfaces::msg::AprilTagDetectionArray)
    CREATE_ROS_TYPE_MONITOR_HELPER(sensor_msgs::msg::CameraInfo)
    CREATE_ROS_TYPE_MONITOR_HELPER(stereo_msgs::msg::DisparityImage)
    CREATE_ROS_TYPE_MONITOR_HELPER(sensor_msgs::msg::Image)
    CREATE_ROS_TYPE_MONITOR_HELPER(sensor_msgs::msg::PointCloud2)
    CREATE_ROS_TYPE_MONITOR_HELPER(sensor_msgs::msg::CompressedImage)
    CREATE_ROS_TYPE_MONITOR_HELPER(isaac_ros_tensor_list_interfaces::msg::TensorList)
    CREATE_ROS_TYPE_MONITOR_HELPER(nav_msgs::msg::OccupancyGrid)
    CREATE_ROS_TYPE_MONITOR_HELPER(vision_msgs::msg::Detection2DArray)
    CREATE_ROS_TYPE_MONITOR_HELPER(vision_msgs::msg::Detection3DArray)

    {
      std::stringstream error_msg;
      error_msg <<
        "[NitrosMonitorNode] Could not identify the monitor subscriber ROS type \"" <<
        ros_type_name.c_str() << "\" was not supported";
      RCLCPP_ERROR(get_logger(), error_msg.str().c_str());
      throw std::runtime_error(error_msg.str().c_str());
    }
  }
}

template<typename ROSMessageType>
void NitrosMonitorNode::CreateROSTypeMonitorSubscriber()
{
  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  std::function<void(std::shared_ptr<ROSMessageType>)> monitor_subscriber_callback =
    std::bind(
    &NitrosMonitorNode::ROSTypeMonitorSubscriberCallback<ROSMessageType>,
    this,
    std::placeholders::_1);

  monitor_sub_ = create_subscription<ROSMessageType>(
    "output",
    ros2_benchmark::kQoS,
    monitor_subscriber_callback,
    sub_options);

  RCLCPP_INFO(
    get_logger(),
    "[NitrosMonitorNode] Created a ROS type monitor subscriber: topic=\"%s\"",
    monitor_sub_->get_topic_name());
}

void NitrosMonitorNode::CreateNitrosMonitorSubscriber()
{
  // Load the extensions needed by the used NITROS format
  RCLCPP_INFO(
    get_logger(),
    "Loading extensions for NITROS format %s",
    monitor_data_format_.c_str());
  nitros_type_manager_->loadExtensions(monitor_data_format_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  std::function<void(nvidia::isaac_ros::nitros::NitrosTypeBase &, const std::string)>
  monitor_subscriber_callback =
    std::bind(
    &NitrosMonitorNode::NitrosTypeMonitorSubscriberCallback,
    this,
    std::placeholders::_1,
    std::placeholders::_2);

  nitros_type_manager_->getFormatCallbacks(monitor_data_format_)
  .createCompatibleSubscriberCallback(
    *this,
    monitor_sub_,
    "output",
    ros2_benchmark::kQoS,
    monitor_subscriber_callback,
    sub_options);

  RCLCPP_INFO(
    get_logger(),
    "[NitrosMonitorNode] Created an NITROS type monitor subscriber: topic=\"%s\"",
    monitor_sub_->get_topic_name());
}

template<typename T>
void NitrosMonitorNode::ROSTypeMonitorSubscriberCallback(const std::shared_ptr<T> msg)
{
  if (revise_timestamps_as_message_ids_) {
    RecordEndTimestamp(msg->header.stamp.sec);
  } else {
    RecordEndTimestampAutoKey();
  }
}

void NitrosMonitorNode::NitrosTypeMonitorSubscriberCallback(
  nvidia::isaac_ros::nitros::NitrosTypeBase & msg_base,
  std::string data_format_name)
{
  (void)data_format_name;

  if (revise_timestamps_as_message_ids_) {
    gxf_result_t code;
    std_msgs::msg::Header ros_header;

    code = nvidia::isaac_ros::nitros::GetTypeAdapterNitrosContext().getEntityTimestamp(
      msg_base.handle, ros_header);
    if (code != GXF_SUCCESS) {
      RCLCPP_ERROR(
        get_logger(),
        "[NitrosMonitorNode] getEntityTimestamp Error");
    }

    RecordEndTimestamp(ros_header.stamp.sec);
  } else {
    RecordEndTimestampAutoKey();
  }
}

}  // namespace isaac_ros_benchmark

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_benchmark::NitrosMonitorNode)
