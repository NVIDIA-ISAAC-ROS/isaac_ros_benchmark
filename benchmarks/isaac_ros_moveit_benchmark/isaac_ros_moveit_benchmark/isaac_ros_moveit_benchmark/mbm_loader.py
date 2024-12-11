# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# The dataset format was derived from Robometrics repository
# https://github.com/fishbotics/robometrics/tree/81e3d1d6
#
# MIT License
#
# Copyright (c) 2023 Adam Fishman, Balakumar Sundaralingam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gzip

import os

from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory

import numpy as np

import yaml

try:
    import geometrout
    _HAS_GEOMETROUT = True
except ImportError:
    _HAS_GEOMETROUT = False


class MBMLoader():

    def __init__(self, robot_group):
        moveit_benchmark_path = os.path.join(
            get_package_share_directory('isaac_ros_moveit_benchmark'), 'datasets')
        self.dataset_path = os.path.join(
            moveit_benchmark_path, f'{robot_group}_mbm_dataset.yaml.gz')

    def structure_problems(self, problem_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert _HAS_GEOMETROUT, 'Optional package geometrout\
                                not installed, so this function is disabled'

        assert set(problem_dict.keys()) == {
            'collision_buffer_ik',
            'goal_ik',
            'goal_pose',
            'obstacles',
            'start',
            'world_frame',
        }
        assert set(problem_dict['obstacles'].keys()) == {'cuboid', 'cylinder'}

        obstacles = {
            'cylinder': {
                k: geometrout.Cylinder(
                    height=v['height'],
                    radius=v['radius'],
                    center=np.array(v['pose'][:3]),
                    quaternion=np.array(v['pose'][3:]),
                )
                for k, v in problem_dict['obstacles']['cylinder'].items()
            },
            'cuboid': {
                k: geometrout.Cuboid(
                    dims=np.array(v['dims']),
                    center=np.array(v['pose'][:3]),
                    quaternion=np.array(v['pose'][3:]),
                )
                for k, v in problem_dict['obstacles']['cuboid'].items()
            },
        }
        return {
            'collision_buffer_ik': problem_dict['collision_buffer_ik'],
            'goal_ik': [np.asarray(ik) for ik in problem_dict['goal_ik']],
            'goal_pose_frame': problem_dict['goal_pose']['frame'],
            'goal_pose': geometrout.SE3(
                pos=np.array(problem_dict['goal_pose']['position_xyz']),
                quaternion=np.array(problem_dict['goal_pose']['quaternion_wxyz']),
            ),
            'obstacles': obstacles,
            'start': np.asarray(problem_dict['start']),
            'world_frame': problem_dict['world_frame'],
        }

    def demo_raw(self) -> Dict[str, Any]:
        with open(self.dataset_path) as f:
            return yaml.safe_load(f)

    def structure_dataset(self, raw_dataset: Dict[str, Any]) -> Dict[str, Any]:
        for problem_set in raw_dataset.values():
            for ii, v in enumerate(problem_set):
                problem_set[ii] = self.structure_problems(v)
        return raw_dataset

    def motion_benchmaker_raw(self) -> Dict[str, Any]:
        with gzip.open(self.dataset_path, 'rt') as f:
            return yaml.safe_load(f)

    def motion_benchmaker(self) -> Dict[str, Any]:
        raw_data = self.motion_benchmaker_raw()
        return self.structure_dataset(raw_data)
