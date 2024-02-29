# Copyright 2022 Lunar Ring. All rights reserved.
# Written by Johannes Stelzer, email stelzer@lunar-ring.ai twitter @j_stelzer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
torch.backends.cudnn.benchmark = False
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import warnings
import datetime
from typing import List, Union
torch.set_grad_enabled(False)
import yaml
import PIL

@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r"""
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Slerp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0:
            First tensor for interpolation
        p1:
            Second tensor for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """

    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'

    # p0 = p0.double()
    # p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1 + epsilon, 1 - epsilon)

    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0 * s0 + p1 * s1

    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()

    return interp


def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0:
            First tensor / np.ndarray for interpolation
        p1:
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float
            Mixing coefficient of interval [0, 1].
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)

    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)

    interp = (1 - fract_mixing) * p0 + fract_mixing * p1

    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)

    return interp


def get_spacing(nmb_points: int, scaling: float):
    """
    Helper function for getting nonlinear spacing between 0 and 1, symmetric around 0.5
    Args:
        nmb_points: int
            Number of points between [0, 1]
        scaling: float
            Higher values will return higher sampling density around 0.5
    """
    if scaling < 1.7:
        return np.linspace(0, 1, nmb_points)
    nmb_points_per_side = nmb_points // 2 + 1
    if np.mod(nmb_points, 2) != 0:  # Uneven case
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side)**scaling / 2 - 0.5)
        right_side = 1 - left_side[::-1][1:]
    else:
        left_side = np.abs(np.linspace(1, 0, nmb_points_per_side)**scaling / 2 - 0.5)[0:-1]
        right_side = 1 - left_side[::-1]
    all_fracts = np.hstack([left_side, right_side])
    return all_fracts


def get_time(resolution=None):
    """
    Helper function returning an nicely formatted time string, e.g. 221117_1620
    """
    if resolution is None:
        resolution = "second"
    if resolution == "day":
        t = time.strftime('%y%m%d', time.localtime())
    elif resolution == "minute":
        t = time.strftime('%y%m%d_%H%M', time.localtime())
    elif resolution == "second":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
    elif resolution == "millisecond":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
        t += "_"
        t += str("{:03d}".format(int(int(datetime.utcnow().strftime('%f')) / 1000)))
    else:
        raise ValueError("bad resolution provided: %s" % resolution)
    return t


def compare_dicts(a, b):
    """
    Compares two dictionaries a and b and returns a dictionary c, with all
    keys,values that have shared keys in a and b but same values in a and b.
    The values of a and b are stacked together in the output.
    Example:
        a = {}; a['bobo'] = 4
        b = {}; b['bobo'] = 5
        c = dict_compare(a,b)
        c = {"bobo",[4,5]}
    """
    c = {}
    for key in a.keys():
        if key in b.keys():
            val_a = a[key]
            val_b = b[key]
            if val_a != val_b:
                c[key] = [val_a, val_b]
    return c


def yml_load(fp_yml, print_fields=False):
    """
    Helper function for loading yaml files
    """
    with open(fp_yml) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
    dict_data = dict(data)
    print("load: loaded {}".format(fp_yml))
    return dict_data


def yml_save(fp_yml, dict_stuff):
    """
    Helper function for saving yaml files
    """
    with open(fp_yml, 'w') as f:
        yaml.dump(dict_stuff, f, sort_keys=False, default_flow_style=False)
    print("yml_save: saved {}".format(fp_yml))
