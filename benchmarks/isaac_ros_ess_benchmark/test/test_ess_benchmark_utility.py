# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Check the ESS graph's monitor registration contract without launching ROS."""

import unittest
from unittest.mock import Mock

from isaac_ros_ess_benchmark import ess_benchmark_utility


class TestEssBenchmarkUtility(unittest.TestCase):
    """Preserve graph diagnostics while selecting final depth for peak search."""

    def create_monitors(self, **kwargs):
        """Capture registrations in the order used to create monitor metadata."""
        utility = Mock()
        utility.add_basic_perf_buffer_monitor.side_effect = lambda *args, **options: args[3]
        nodes = ess_benchmark_utility.create_ess_depth_graph_monitors(
            utility, 'benchmark', **kwargs)
        return utility.add_basic_perf_buffer_monitor.call_args_list, nodes

    def test_depth_is_first_registered_and_all_monitors_remain(self):
        """Search observes Depth even though camera-info nodes load first."""
        calls, nodes = self.create_monitors()
        self.assertEqual(calls[0].args[3:5], ('depth', 'Depth'))
        self.assertEqual(calls[0].kwargs['monitor_name'], 'DepthMonitorNode')
        topics = [call.args[3] for call in calls]
        self.assertEqual(len(topics), 10)
        self.assertEqual(len(set(topics)), 10)
        self.assertCountEqual(nodes, topics)
        self.assertTrue(all('camera_info' in topic for topic in nodes[:4]))
        self.assertEqual(nodes[4], 'depth')

    def test_topic_filters_keep_only_requested_monitors(self):
        """Depth is primary when selected; omitting it adds no hidden monitor."""
        cases = [
            ([], []),
            (['depth'], ['depth']),
            (['left/image_rect', 'depth'], ['depth', 'left/image_rect']),
            (['disparity', 'left/image_rect'], ['left/image_rect', 'disparity']),
            (['left/camera_info_rect', 'depth'], ['depth', 'left/camera_info_rect']),
        ]
        for topics, expected in cases:
            with self.subTest(topics=topics):
                calls, nodes = self.create_monitors(monitor_topics=topics)
                self.assertEqual([call.args[3] for call in calls], expected)
                self.assertCountEqual(nodes, expected)


if __name__ == '__main__':
    unittest.main()
