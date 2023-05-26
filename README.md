# Isaac ROS Benchmark

<div align="center"><img alt="ROS 2 Turtlebot launch" src="resources/r2b_turtlebot_takeoff.gif" width="600"/></div>

## Overview

This package builds upon the [`ros2_benchmark`](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) to provide configurations to benchmark Isaac ROS graphs. Performance results that measure Isaac ROS for throughput, latency, and utilization enable robotics developers to make informed decisions when designing real-time robotics applications. The Isaac ROS performance results can be independently verified, as the method, configuration, and data input used for benchmarking are provided.

The `ros2_benchmark` playback node plug-in for type adaptation and negotiation is provided for [NITROS](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros), which optimizes the performance of message transport costs through [RCL](https://github.com/ros2/rclcpp) with GPU accelerated graphs of nodes.

The datasets for benchmarking are explicitly not downloaded by default. To pull down the standardized benchmark datasets, refer to the [`ros2_benchmark` Dataset](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark#datasets) section.

## Performance Results

For a table of the latest performance results for Isaac ROS see this [performance summary](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/performance-summary.md).

## Table of Contents

- [Isaac ROS Benchmark](#isaac-ros-benchmark)
  - [Overview](#overview)
  - [Performance Results](#performance-results)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Benchmark Configurations](#benchmark-configurations)
    - [Preprocessors](#preprocessors)
    - [Graph Under Test](#graph-under-test)
    - [Required Packages](#required-packages)
    - [Required Datasets](#required-datasets)
    - [Required Models](#required-models)
    - [List of Isaac ROS Benchmarks](#list-of-isaac-ros-benchmarks)
    - [Results](#results)
  - [Profiling](#profiling)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2023-04-05: Initial release.

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

| Platform | Hardware                                                                                                                                                                                               | Software                                                                                                         | Notes                                                                                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)<br>[Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                   | For best performance, ensure that [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                             | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/)<br>[CUDA 11.8+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note**: All Isaac ROS Quickstarts, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

Follow the steps below to run a sample benchmark for measuring performance of an Isaac ROS AprilTag node with `ros2_benchmark`. This process can also be used to benchmark the other Isaac ROS nodes, and the `ros2_benchmark` framework more generally supports benchmarking arbitrary graphs of ROS 2 nodes.

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).

2. Clone this repository and its its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src &&
      git clone https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark && \
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark && \
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common && \
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros && \
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline && \
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag
    ```

3. Pull down `r2b Dataset 2023` by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark/blob/main/README.md#datasets) or fetch just the rosbag used in this Quickstart with the following command.

    ```bash
    mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/datasets/r2b_dataset/r2b_storage && \
      cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/datasets/r2b_dataset/r2b_storage && \
      wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/1/files/r2b_storage/metadata.yaml' && \
      wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/1/files/r2b_storage/r2b_storage_0.db3'
    ```

4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

5. Inside the container, build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

6. (Optional) Run tests to verify complete and correct installation:

    ```bash
    colcon test --executor sequential
    ```

7. Start the Isaac ROS AprilTag benchmark:

    ```bash
    launch_test src/isaac_ros_benchmark/scripts/isaac_ros_apriltag_node.py
    ```

8. Once the benchmark is finished, the final performance measurements are displayed in the terminal.

   Additionally, the final results and benchmark metadata (e.g., system information, benchmark configurations) are also exported as a JSON file.

## Benchmark Configurations

Performance measurement for graphs of nodes requires a ROS 2 launch file to launch the benchmark and an input YAML file specifying the configuration information.

Each Isaac ROS Benchmark launch file details its own benchmark configuration via a comment near the top of the file. For example, the following comment is included immediately following the license header in `isaac_ros_benchmark/scripts/isaac_ros_apriltag_node.py`:

```python
"""
Performance test for Isaac ROS AprilTagNode.

The graph consists of the following:
- Preprocessors:
    1. PrepResizeNode: resizes images to HD
- Graph under Test:
    1. AprilTagNode: detects Apriltags

Required:
- Packages:
    - isaac_ros_image_proc
    - isaac_ros_apriltag
- Datasets:
    - assets/datasets/r2b_dataset/r2b_storage
"""
```

Each section of this comment is explained in further detail below.

### Preprocessors

In some cases, the desired input sequence contains data that is not yet in the appropriate format to be received by the ROS 2 graph under test. For example, in the case of `isaac_ros_apriltag_node.py`, the input dataset's images must first be resized into HD resolution before being passed into the AprilTag detecting node.

The preprocessing nodes allow for these types of data transformations to be executed before the critical timing section of the benchmark begins, ensuring that there is no undesired penalty in performance.

### Graph Under Test

The graph under test refers to the core selection of ROS 2 nodes whose performance is to be measured in this specific benchmark. For example, in the case of `isaac_ros_apriltag_node.py`, only the Isaac ROS AprilTag detecting node is included under the graph under test.

By contrast, the `isaac_ros_apriltag_graph.py` benchmark includes multiple nodes in its graph under test:

```python
"""
[...]
- Graph under Test:
    1. RectifyNode: rectifies images
    2. AprilTagNode: detects Apriltags
[...]
"""
```

The Isaac ROS Benchmark collection of benchmark scripts includes both individual node and composite graphs under test. Node-specific benchmarks, identified by the `_node` suffix, showcase the absolute maximum performance possible when a node is run in isolation. Larger graph benchmarks, identified by the `_graph` suffix, present performance in a more typical use case.

### Required Packages

Each benchmark in the Isaac ROS Benchmark collection includes a different selection of nodes as preprocessors or components of the graph under test. Consequently, each benchmark requies its own specific subset of the Isaac ROS suite of packages in order to successfully run.

For example, the `isaac_ros_apriltag_graph.py` benchmark directly depends on the `isaac_ros_image_proc` and `isaac_ros_apriltag` packages. These packages, along with their own recursive dependencies, must be properly built and sourced prior to running the benchmark.

### Required Datasets

The Isaac ROS Benchmark scripts use the standard r2b Dataset 2023 collection of input data. Before running any benchmarks, the input datasets must be downloaded by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark/blob/main/README.md#datasets).

### Required Models

Some of the benchmark graphs require loading model files. Models used by a benchmark graph are listed in the benchmark script's header. By default models are expected to be accessible by a benchmark script under `~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models`.

Before benchmarking a node that contain a DNN, the DNN must first be downloaded and converted to a `.plan` file for the host system, using the instructions provided in the table below:

| Model                                                                                                                                  | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Bi3D](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation#model-preparation)                                         | `mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/bi3d && cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/bi3d && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/featnet.onnx' && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/segnet.onnx'`                                                                                                                                                                                                                                                                                                                                                                                                               |
| [ESS](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/dnn_stereo_disparity)                                              | `mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/ess && cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/ess && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity/versions/1.0.0/files/ess.etlt'`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [DOPE](https://github.com/NVlabs/Deep_Object_Pose) [Ketchup](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg) | `mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/ketchup && cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/ketchup`<br>Download `Ketchup.pth` model from [here](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg) to the current directory.<br>Start the Isaac ROS Docker container before running the next step: `~/workspaces/isaac_ros-dev/scripts/run_dev.sh && python3 /workspaces/isaac_ros-dev/src/isaac_ros_pose_estimation/isaac_ros_dope/scripts/dope_converter.py --format onnx --input Ketchup.pth --output ketchup.onnx`<br>Create a file `config.pbtxt` in the current directory by using the configurations provided in setup 4 in [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/blob/main/docs/dope-triton.md). |
| [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)                                                     | `mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/peoplenet && cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/peoplenet && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.etlt' && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.txt' && wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/labels.txt'`                                                                                                                                                                                                                                     |
| [PeopleSemSegNet ShuffleSeg](https://ngc.nvidia.com/catalog/models/nvidia:tao:peoplesemsegnet)                                         | `mkdir -p ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/peoplesemsegnet_shuffleseg && cd ~/workspaces/isaac_ros-dev/src/ros2_benchmark/assets/models/peoplesemsegnet_shuffleseg && wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_etlt.etlt && wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplesemsegnet/versions/deployable_shuffleseg_unet_v1.0/files/peoplesemsegnet_shuffleseg_cache.txt && cp ~/workspaces/isaac_ros-dev/src/isaac_ros_dnn_inference/resources/peoplesemsegnet_shuffleseg_config.pbtxt config.pbtxt`                                                                                                                                                                  |

### List of Isaac ROS Benchmarks

> **Note**: Prior to running any of the following benchmark scripts, please ensure that your environment satisfies all prerequisites:
>
> - [ ] All required datasets have been downloaded per the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark/blob/main/README.md#datasets)
> - [ ] For DNN-based benchmarks, all required DNNs have been prepared per the instructions [here](#required-models)
> - [ ] All required packages have been downloaded, built, and sourced
<!-- Split blockquote -->
> **Note**: We use the naming convention `_node` to represent a graph under test that contains a single node (for example, `stereo_image_proc_node.py`) and `_graph` to represent a graph of multiple nodes (for example, `stereo_image_proc_graph.py`).

| Name                                    | Description                                                                                    | Dataset Sequence                              | Launch Command                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------- |
| AprilTag Node                           | Detect AprilTags                                                                               | `r2b_storage`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_apriltag_node.py`            |
| AprilTag Graph                          | Rectify image and detect AprilTags                                                             | `r2b_storage`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_apriltag_graph.py`           |
| Freespace Segmentation Node             | Project freespace onto occupancy grid                                                          | `r2b_lounge`                                  | `launch_test isaac_ros_benchmark/scripts/isaac_ros_bi3d_fs_node.py`             |
| Freespace Segmentation Graph            | Create proximity segmentation disparity image and project freespace onto occupancy grid        | `r2b_lounge`                                  | `launch_test isaac_ros_benchmark/scripts/isaac_ros_bi3d_fs_graph.py`            |
| Proximity Segmentation Node             | Create proximity segmentation disparity image                                                  | `r2b_lounge`                                  | `launch_test isaac_ros_benchmark/scripts/isaac_ros_bi3d_node.py`                |
| Centerpose Pose Estimation Graph        | Encode image, run CenterPose model inference on TensorRT, decode output tensor as marker array | `r2b_storage`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_centerpose_graph.py`         |
| DetectNet Object Detection Graph        | Encode image, run PeopleNet on Triton, decode output tensor as detection array                 | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_detectnet_graph.py`          |
| Stereo Disparity Node                   | Create stereo disparity image                                                                  | `r2b_datacenter`                              | `launch_test isaac_ros_benchmark/scripts/isaac_ros_disparity_node.py`           |
| Stereo Disparity Graph                  | Create stereo disparity image, convert disparity image to point cloud                          | `r2b_datacenter`                              | `launch_test isaac_ros_benchmark/scripts/isaac_ros_disparity_graph.py`          |
| DNN Image Encoder Node                  | Encode image as resized, normalized tensor                                                     | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_dnn_image_encoder_node.py`   |
| DOPE Pose Estimation Graph              | Encode image, run DOPE on TensorRT, decode output tensor as pose array                         | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_dope_graph.py`               |
| DNN Stereo Disparity Node               | Create ESS-inferred stereo disparity image                                                     | `r2b_hideaway`                                | `launch_test isaac_ros_benchmark/scripts/isaac_ros_ess_node.py`                 |
| DNN Stereo Disparity Graph              | Create ESS-inferred stereo disparity image, convert disparity image to point cloud             | `r2b_hideaway`                                | `launch_test isaac_ros_benchmark/scripts/isaac_ros_ess_graph.py`                |
| Occupancy Grid Localizer Node           | Estimate pose relative to map                                                                  | `r2b_storage`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_grid_localizer_node.py`      |
| H264 Decoder Node                       | Decode compressed image                                                                        | `r2b_compressed_image`[^r2b_compressed_image] | `launch_test isaac_ros_benchmark/scripts/isaac_ros_h264_decoder_node.py`        |
| H264 Encoder Node </br> I-frame Support | Encode compressed image (I-frame)                                                              | `r2b_mezzanine`                               | `launch_test isaac_ros_benchmark/scripts/isaac_ros_h264_encoder_iframe_node.py` |
| H264 Encoder Node </br> P-frame Support | Encode compressed image (P-frame)                                                              | `r2b_mezzanine`                               | `launch_test isaac_ros_benchmark/scripts/isaac_ros_h264_encoder_pframe_node.py` |
| Nvblox Node                             | Generate colorized 3D mesh                                                                     | `r2b_hideaway`                                | `launch_test isaac_ros_benchmark/scripts/isaac_ros_nvblox_node.py`              |
| Rectify Node                            | Rectify image                                                                                  | `r2b_storage`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_rectify_node.py`             |
| TensorRT Node </br> PeopleSemSegNet     | Run PeopleSemSegNet inference on TensorRT                                                      | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_tensor_rt_ps_node.py`        |
| TensorRT Node </br> DOPE                | Run DOPE inference on TensorRT                                                                 | `r2b_hope`                                    | `launch_test isaac_ros_benchmark/scripts/isaac_ros_tensor_rt_dope_node.py`      |
| Triton Node </br> PeopleSemSegNet       | Run PeopleSemSegNet inference on Triton                                                        | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_triton_ps_node.py`           |
| Triton Node </br> DOPE                  | Run DOPE inference on Triton                                                                   | `r2b_hope`                                    | `launch_test isaac_ros_benchmark/scripts/isaac_ros_triton_dope_node.py`         |
| UNet Graph                              | Encode image, run PeopleSemSegNet on TensorRT, decode output tensor as segmentation masks      | `r2b_hallway`                                 | `launch_test isaac_ros_benchmark/scripts/isaac_ros_unet_graph.py`               |
| Visual SLAM Node                        | Perform stereo visual simultaneous localization and mapping                                    | `r2b_cafe`                                    | `launch_test isaac_ros_benchmark/scripts/isaac_ros_visual_slam_node.py`         |

[^r2b_compressed_image]: `r2b_compressed_image` was generated from `r2b_mezzanine` by recapturing replayed data through `isaac_ros_h264_encoder` on a Jetson using this [launch graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_compression/blob/main/isaac_ros_h264_encoder/launch/isaac_ros_encode_r2b_rosbag.launch.py).

### Results

The [Isaac ROS Performance Summary](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/performance-summary.md) provides an overview of the benchmark results for the Isaac ROS configurations detailed above, along with links to each of the results in JSON format.

## Profiling

When seeking to optimize performance, profiling is often used to gain deep insight into the call stack and to identify where processing time is spent in functions. [`ros2_tracing`](https://github.com/ros2/ros2_tracing) provides a tracing instrumentation to better understand performance on a CPU, but lacks information on GPU acceleration.

[Nsight Systems](https://developer.nvidia.com/nsight-systems) is a freely-available tool that provides tracing instrumentation for CPU, GPU, and other SOC (system-on-chip) hardware accelerators on both `aarch64` and `x86_64` platforms. The Isaac ROS team uses this tooling internally to profile Isaac ROS graphs, to optimize individual node-level computation, and to improve synchronization between heterogenous computing hardware. These tools allow for before-and-after testing to inspect profile differences with the benchmark tooling.

Profiling hooks for Nsight Systems have been integrated in [`ros2_benchmark`](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) scripts for rich annotations. The following commands show an example of how to use Nsight Systems to profile a benchmark a fiducial detection graph built with [Isaac ROS AprilTag](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag):

```bash
launch_test src/isaac_ros_benchmark/isaac_ros_benchmark/scripts/isaac_ros_apriltag_node.py enable_nsys:=true nsys_profile_name:=isaac_ros_apriltag_profile
```

| Nsys Parameters      | Default                                 | Description                     |
| -------------------- | --------------------------------------- | ------------------------------- |
| `enable_nsys`        | `false`                                 | Enable nsys or not              |
| `nsys_profile_flags` | `--trace=osrt,nvtx,cuda`                | Flags passed to nsys            |
| `nsys_profile_name`  | `profile_{machine_type}_{current_time}` | Nsys profiling output file name |

## Troubleshooting

### Isaac ROS Troubleshooting

For solutions to problems with Isaac ROS, please check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

## Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2023-04-05 | Initial release |
