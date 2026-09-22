# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utilities for Isaac ROS tensors."""
from typing import List

from isaac_ros_tensor_msgs.msg import TensorList

import numpy as np
from tensor_msgs.msg import ExperimentalTensor


class TensorUtility:
    """Class for tensor utilities."""

    LEGACY_TO_DLPACK_DTYPE = {
        1: (0, 8, 1),
        2: (1, 8, 1),
        3: (0, 16, 1),
        4: (1, 16, 1),
        5: (0, 32, 1),
        6: (1, 32, 1),
        7: (0, 64, 1),
        8: (1, 64, 1),
        9: (2, 32, 1),
        10: (2, 64, 1),
    }

    @staticmethod
    def load_random_tensors(shape: List[int],
                            datatype: int = 9,
                            stride: int = 4,
                            name: str = 'input',
                            batch: int = 1,
                            duplicate: bool = False) -> List[TensorList]:
        """
        Generate random tensors.

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        datatype : int
            Tensor type
        stride : int
            Memory stride of that tensor type
        name: str
            Tensor name
        batch:
            Batch size of random tensors
        duplicate:
            Reuse the same tensor

        Returns
        -------
        List[TensorList]
            Random tensors

        """
        data_length = stride * np.prod(shape)

        random_tensors = []
        for i in range(batch):
            tensor_list = TensorList()
            tensor = ExperimentalTensor()
            dtype = TensorUtility.LEGACY_TO_DLPACK_DTYPE.get(datatype)
            if dtype is None:
                raise ValueError(f'Unsupported tensor datatype: {datatype}')
            tensor.dtype_code, tensor.dtype_bits, tensor.dtype_lanes = dtype
            tensor.shape = [int(dim) for dim in shape]
            tensor.strides = []
            tensor.byte_offset = 0
            tensor.data = np.random.randint(256, size=data_length).tolist()

            tensor_list.names = [name]
            tensor_list.tensors = [tensor]

            if i == 0 and duplicate:
                random_tensors = [tensor_list]*batch
                break
            else:
                random_tensors.append(tensor_list)

        return random_tensors
