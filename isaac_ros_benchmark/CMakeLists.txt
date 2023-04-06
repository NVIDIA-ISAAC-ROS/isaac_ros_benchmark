# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23.2)
project(isaac_ros_benchmark LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_auto_find_build_dependencies()
ament_python_install_package(${PROJECT_NAME})

# NVTX
option(USE_NVTX "Enable NVTX markers for improved profiling (if available)" ON)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    link_libraries("nvToolsExt")
endif()

# NitrosPlaybackNode
ament_auto_add_library(nitros_playback_node SHARED src/nitros_playback_node.cpp)
rclcpp_components_register_nodes(nitros_playback_node "isaac_ros_benchmark::NitrosPlaybackNode")
set(node_plugins "${node_plugins}isaac_ros_benchmark::NitrosPlaybackNode;$<TARGET_FILE:nitros_playback_node>\n")

# NitrosMonitorNode
ament_auto_add_library(nitros_monitor_node SHARED src/nitros_monitor_node.cpp)
rclcpp_components_register_nodes(nitros_monitor_node "isaac_ros_benchmark::NitrosMonitorNode")
set(node_plugins "${node_plugins}isaac_ros_benchmark::NitrosMonitorNode;$<TARGET_FILE:nitros_monitor_node>\n")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Ignore copyright notices since we use custom NVIDIA Isaac ROS Software License
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
  find_package(launch_testing_ament_cmake REQUIRED)

  add_launch_test(test/nitros_playback_node_pol.py TIMEOUT "15")
  add_launch_test(test/nitros_monitor_node_pol.py TIMEOUT "15")
endif()

ament_auto_package(INSTALL_TO_SHARE config)
