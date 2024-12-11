# Isaac ROS Moveit Benchmarking Guide

This guide outlines the steps to build the workspaces and run the benchmark commands for different robots using cuMotion and OMPL planners.

## Build Workspaces

Use the following commands to build the workspaces for UR5 and Franka robots with cuMotion and OMPL planners. Ensure you are in the correct workspace directory before running these commands.

### UR5 cuMotion Benchmark

```bash
colcon build --symlink-install --packages-up-to isaac_ros_ur5_cumotion_benchmark
```

### UR5 OMPL Benchmark

```bash
colcon build --symlink-install --packages-up-to isaac_ros_ur5_ompl_benchmark
``` 

### Franka cuMotion Benchmark

```bash
colcon build --symlink-install --packages-up-to isaac_ros_franka_cumotion_benchmark
```

### Franka OMPL Benchmark

```bash
colcon build --symlink-install --packages-up-to isaac_ros_franka_ompl_benchmark
```

## Run Commands

```bash
launch_test $(ros2 pkg prefix isaac_ros_ur5_cumotion_benchmark)/share/isaac_ros_ur5_cumotion_benchmark/ur5_cumotion_benchmark_scripts/ur5_cumotion_mbm.py


launch_test $(ros2 pkg prefix isaac_ros_ur5_ompl_benchmark)/share/isaac_ros_ur5_ompl_benchmark/ur5_ompl_benchmark_scripts/ur5_ompl_mbm.py


launch_test $(ros2 pkg prefix isaac_ros_franka_cumotion_benchmark)/share/isaac_ros_franka_cumotion_benchmark/franka_cumotion_benchmark_scripts/franka_cumotion_mbm.py


launch_test $(ros2 pkg prefix isaac_ros_franka_ompl_benchmark)/share/isaac_ros_franka_ompl_benchmark/franka_ompl_benchmark_scripts/franka_ompl_mbm.py 
```