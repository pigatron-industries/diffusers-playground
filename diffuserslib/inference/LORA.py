from safetensors.torch import load_file
import torch
import os

LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'

class LORA:

    @classmethod
    def from_file(cls, lora_path, device = "cuda", weight = 1):
        if(os.path.isdir(lora_path)):
            return DiffusersLORA.from_file(lora_path)
        else:
            return StableDiffusionLORA.from_file(lora_path, device, weight)


class DiffusersLORA(LORA):

    def __init__(self, lora_path):
        self.lora_path = lora_path

    @classmethod
    def from_file(cls, lora_path):
        return cls(lora_path)
    
    def add_to_model(self, pipeline, weight = 1):
        pipeline.unet.load_attn_procs(self.lora_path)
    


class StableDiffusionLORA(LORA):

    def __init__(self, state_dict, weight = 1):
        self.state_dict = state_dict
        self.weight = weight

    @classmethod
    def from_file(cls, lora_path, device, weight = 1):
        state_dict = load_file(lora_path, device=device)
        return cls(state_dict, weight)

    def add_to_model(self, pipeline):
        print("weight")
        print(self.weight)
        visited = []
        for key in self.state_dict:

            # as we have set the alpha beforehand, so just skip
            if '.alpha' in key or key in visited:
                continue

            if 'text' in key:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
                curr_layer = pipeline.text_encoder
            else:
                layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
                curr_layer = pipeline.unet

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
                        temp_name += '_'+layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # org_forward(x) + lora_up(lora_down(x)) * multiplier
            pair_keys = []
            if 'lora_down' in key:
                pair_keys.append(key.replace('lora_down', 'lora_up'))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace('lora_up', 'lora_down'))

            # update weight
            if len(self.state_dict[pair_keys[0]].shape) == 4:
                weight_up = self.state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = self.state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data += self.weight * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = self.state_dict[pair_keys[0]].to(torch.float32)
                weight_down = self.state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += self.weight * torch.mm(weight_up, weight_down)

            # update visited list
            for item in pair_keys:
                visited.append(item)