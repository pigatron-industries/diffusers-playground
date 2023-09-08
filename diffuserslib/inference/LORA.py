from safetensors.torch import load_file
from collections import defaultdict
import torch
import os

LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'


class LORA:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    @classmethod
    def from_file(cls, name, path):
        if (path.endswith('.bin')):
            return DiffusersLORA(name, path)
        else:
            return StableDiffusionLORA(name, path)
        
    def add_to_model(self, pipeline, weight = 1, device="cuda"):
        pass


class DiffusersLORA(LORA):
    def __init__(self, name, path):
        super().__init__(name, path)
    
    def add_to_model(self, pipeline, weight = 1, device="cuda"):
        lora = torch.load(self.path)
        for key in lora.keys():
            lora[key] = lora[key] * weight
        pipeline.load_lora_weights(lora)


class StableDiffusionLORA(LORA):
    def __init__(self, name, path):
        super().__init__(name, path)

    def add_to_model(self, pipeline, weight = 1, device="cuda"):
        state_dict = load_file(self.path, device=device)
        state_dict = load_file(self.path, device="cpu")
        for key, value in state_dict.items():
            state_dict[key] = value.to(torch.float16).to(device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if LORA_PREFIX_TEXT_ENCODER in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = pipeline.text_encoder
            elif LORA_PREFIX_UNET in layer:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet
            else:
                continue

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight']
            weight_down = elems['lora_down.weight']
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += weight * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += weight * alpha * torch.mm(weight_up, weight_down)