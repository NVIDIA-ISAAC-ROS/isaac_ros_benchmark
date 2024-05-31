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

import os
import subprocess
from typing import List


class Converter:

    def __init__(self, name: str, executable: str):
        self._name = name
        self._executable = executable

    def __call__(self, args: List[str]):
        print('Running command:\n' + ' '.join([self._executable] + args))
        result = subprocess.run(
            [self._executable] + args,
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise Exception(
                f'{self._name} failed to convert with status: {result.returncode}.\n'
                f'stderr:\n' + result.stderr.decode('utf-8')
            )


class TaoConverter(Converter):

    def __init__(self):
        super(TaoConverter, self).__init__(
            'tao-converter',
            '/opt/nvidia/tao/tao-converter'
        )


class TRTConverter(Converter):

    def __init__(self):
        super(TRTConverter, self).__init__(
            'trt-converter',
            '/usr/src/tensorrt/bin/trtexec'
        )
