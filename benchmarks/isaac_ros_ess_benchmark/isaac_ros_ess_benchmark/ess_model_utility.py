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

import os
import time

from isaac_ros_ess.engine_generator import ESSEngineGenerator
from ros2_benchmark import ROS2BenchmarkTest

# ESS mode paths
FULL_ESS_MODEL_FILE_NAME = 'ess/ess.etlt'
FULL_ESS_ENGINE_FILE_PATH = 'ess/ess.engine'
LIGHT_ESS_MODEL_FILE_NAME = 'ess/light_ess.etlt'
LIGHT_ESS_ENGINE_FILE_PATH = 'ess/light_ess.engine'

# ESS mode resolution
FULL_ESS_RESOLUTION = {'width': 960, 'height': 576}
LIGHT_ESS_RESOLUTION = {'width': 480, 'height': 288}


def get_mode_resolution(ess_model_type):
    if ess_model_type == 'full':
        return FULL_ESS_RESOLUTION
    elif ess_model_type == 'light':
        return LIGHT_ESS_RESOLUTION
    else:
        raise ValueError('Unrecognized ESS model type: {}'.format(ess_model_type))


def get_model_root(asset_models_root=None):
    if asset_models_root is None:
        return os.path.join(ROS2BenchmarkTest.get_assets_root_path(), 'models')
    else:
        return asset_models_root


def get_model_paths(ess_model_type, asset_models_root=None):
    model_root = get_model_root(asset_models_root)

    if ess_model_type == 'full':
        ess_model_path = os.path.join(model_root, FULL_ESS_MODEL_FILE_NAME)
        ess_engine_path = os.path.join(model_root, FULL_ESS_ENGINE_FILE_PATH)
    elif ess_model_type == 'light':
        ess_model_path = os.path.join(model_root, LIGHT_ESS_MODEL_FILE_NAME)
        ess_engine_path = os.path.join(model_root, LIGHT_ESS_ENGINE_FILE_PATH)
    else:
        raise ValueError('Unrecognized ESS model type: {}'.format(ess_model_type))

    return ess_model_path, ess_engine_path


def generate_ess_engine_file(ess_model_type, asset_models_root=None):
    ess_model_path, ess_engine_path = get_model_paths(ess_model_type, asset_models_root)

    # Generate engine file using tao-converter
    if not os.path.isfile(ess_engine_path):
        print(f'Generating engine file for {ess_model_type} ESS model...')
        gen = ESSEngineGenerator(etlt_model=ess_model_path)
        start_time = time.time()
        gen.generate()
        print('ESS model engine file generation was finished '
              f'(took {(time.time() - start_time)}s)')
    else:
        print(f'An ESS engine file was found at "{ess_engine_path}"')


def is_ess_engine_file_generated(ess_model_type='full', asset_models_root=None):
    _, ess_engine_path = get_model_paths(ess_model_type, asset_models_root)
    return os.path.isfile(ess_engine_path)
