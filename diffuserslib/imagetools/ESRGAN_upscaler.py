import os
from functools import partial

import numpy as np
import torch
from PIL import Image

from .ESRGAN_arch import *
from ..ImageUtils import tiledImageProcessor


class ESRGANUpscaler():
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = load_model(model_path, device)

    def upscale(self, inputimage, scale=4):
        img = np.array(inputimage)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img)
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = 255. * np.moveaxis(output, 0, 2)
        output = output.astype(np.uint8)
        output = output[:, :, ::-1]
        outimage = Image.fromarray(output, 'RGB')
        if (scale != 4):
            outimage = outimage.resize((inputimage.width*scale, inputimage.height*scale))
        return outimage
    
    def upscaleTiled(self, img, scale=4, tilewidth=768, tileheight=768, overlap=64):
        # return img
        return tiledImageProcessor(partial(self.upscale, scale=scale), img, scale=scale,  tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, reduceEdges=True)


    def __call__(self, img):
        return self.upscale(img)

    def __del__(self):
        del self.model


def mod2normal(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    if 'conv_first.weight' in state_dict:
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict


def resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        re8x = 0
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if "rdb" in k:
                ori_k = k.replace('body.', 'model.1.sub.')
                ori_k = ori_k.replace('.rdb', '.RDB')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
        crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
        crt_net['model.3.weight'] = state_dict['conv_up1.weight']
        crt_net['model.3.bias'] = state_dict['conv_up1.bias']
        crt_net['model.6.weight'] = state_dict['conv_up2.weight']
        crt_net['model.6.bias'] = state_dict['conv_up2.bias']

        if 'conv_up3.weight' in state_dict:
            # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
            re8x = 3
            crt_net['model.9.weight'] = state_dict['conv_up3.weight']
            crt_net['model.9.bias'] = state_dict['conv_up3.bias']

        crt_net[f'model.{8+re8x}.weight'] = state_dict['conv_hr.weight']
        crt_net[f'model.{8+re8x}.bias'] = state_dict['conv_hr.bias']
        crt_net[f'model.{10+re8x}.weight'] = state_dict['conv_last.weight']
        crt_net[f'model.{10+re8x}.bias'] = state_dict['conv_last.bias']

        state_dict = crt_net
    return state_dict


def infer_params(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if (part_num > scalemin
                and parts[0] == "model"
                and parts[2] == "weight"):
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2 ** scale2x

    return in_nc, out_nc, nf, nb, plus, scale


def load_model(filename: str, device = "cuda"):
    if not os.path.exists(filename) or filename is None:
        print(f"Unable to load {filename}")
        return None

    state_dict = torch.load(filename, map_location='cpu' if device == 'mps' else None)

    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]
        num_conv = 16 if "realesr-animevideov3" in filename else 32
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)

    if "body.0.rdb1.conv1.weight" in state_dict and "conv_first.weight" in state_dict:
        nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
        state_dict = resrgan2normal(state_dict, nb)
    elif "conv_first.weight" in state_dict:
        state_dict = mod2normal(state_dict)
    elif "model.0.weight" not in state_dict:
        raise Exception("The file is not a recognized ESRGAN model.")

    in_nc, out_nc, nf, nb, plus, mscale = infer_params(state_dict)

    model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
    model.load_state_dict(state_dict)
    model.eval()

    return model.to(device)

