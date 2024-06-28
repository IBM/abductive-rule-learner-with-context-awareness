#
# Copyright (c) IBM Corp. 2024
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

from arlc.utils.raven.scene import SceneEngine


def get_env(env_name, device, **kwargs):
    if env_name == "center_single":
        return CenterSingle(device, **kwargs)
    if env_name == "distribute_four":
        return DistributeFour(device, **kwargs)
    if env_name == "distribute_nine":
        return DistributeNine(device, **kwargs)
    if env_name == "in_center_single_out_center_single":
        return InCenterSingleOutCenterSingle(device, **kwargs)
    if env_name == "in_distribute_four_out_center_single":
        return InDistributeFourOutCenterSingle(device, **kwargs)
    if env_name == "left_center_single_right_center_single":
        return LeftCenterSingleRightCenterSingle(device, **kwargs)
    if env_name == "up_center_single_down_center_single":
        return UpCenterSingleDownCenterSingle(device, **kwargs)
    return None


class GeneralEnv(object):
    def __init__(self, num_slots, device, **kwargs):
        self.num_slots = num_slots
        self.device = device
        self.scene_engine = SceneEngine(self.num_slots, device)

    def prepare(self, model_output):
        return self.scene_engine.compute_scene_prob(*model_output)


class CenterSingle(GeneralEnv):
    def __init__(self, device, **kwargs):
        super(CenterSingle, self).__init__(1, device, **kwargs)


class DistributeFour(GeneralEnv):
    def __init__(self, device, **kwargs):
        super(DistributeFour, self).__init__(4, device, **kwargs)


class DistributeNine(GeneralEnv):
    def __init__(self, device, **kwargs):
        super(DistributeNine, self).__init__(9, device, **kwargs)


class OutCenterSingle(GeneralEnv):
    def __init__(self, device, **kwargs):
        super(OutCenterSingle, self).__init__(1, device, **kwargs)


class InCenterSingleOutCenterSingle(object):
    def __init__(self, device, **kwargs):
        self.in_center_single = CenterSingle(device, **kwargs)
        self.out_center_single = OutCenterSingle(device, **kwargs)

    def prepare(self, model_output):
        in_component = []
        out_component = []
        for element in model_output:
            in_component.append(element[:, :, 1:, :])
            out_component.append(element[:, :, :1, :])
        in_scene_prob, in_scene_logprob = self.in_center_single.prepare(in_component)
        out_scene_prob, out_scene_logprob = self.out_center_single.prepare(out_component)
        return (in_scene_prob, out_scene_prob), (in_scene_logprob, out_scene_logprob)


class InDistributeFourOutCenterSingle(object):
    def __init__(self, device, **kwargs):
        self.in_distribute_four = DistributeFour(device, **kwargs)
        self.out_center_single = OutCenterSingle(device, **kwargs)

    def prepare(self, model_output):
        in_component = []
        out_component = []
        for element in model_output:
            in_component.append(element[:, :, 1:5, :])
            out_component.append(element[:, :, :1, :])
        in_scene_prob, in_scene_logprob = self.in_distribute_four.prepare(in_component)
        out_scene_prob, out_scene_logprob = self.out_center_single.prepare(out_component)
        return (in_scene_prob, out_scene_prob), (in_scene_logprob, out_scene_logprob)


class LeftCenterSingleRightCenterSingle(object):
    def __init__(self, device, **kwargs):
        self.left_center_single = CenterSingle(device, **kwargs)
        self.right_center_single = CenterSingle(device, **kwargs)

    def prepare(self, model_output):
        left_component = []
        right_component = []
        for element in model_output:
            left_component.append(element[:, :, :1, :])
            right_component.append(element[:, :, 1:, :])
        left_scene_prob, left_scene_logprob = self.left_center_single.prepare(left_component)
        right_scene_prob, right_scene_logprob = self.right_center_single.prepare(right_component)
        return (left_scene_prob, right_scene_prob), (left_scene_logprob, right_scene_logprob)


class UpCenterSingleDownCenterSingle(object):
    def __init__(self, device, **kwargs):
        self.up_center_single = CenterSingle(device, **kwargs)
        self.down_center_single = CenterSingle(device, **kwargs)

    def prepare(self, model_output):
        up_component = []
        down_component = []
        for element in model_output:
            up_component.append(element[:, :, :1, :])
            down_component.append(element[:, :, 1:, :])
        up_scene_prob, up_scene_logprob = self.up_center_single.prepare(up_component)
        down_scene_prob, down_scene_logprob = self.down_center_single.prepare(down_component)
        return (up_scene_prob, down_scene_prob), (up_scene_logprob, down_scene_logprob)
