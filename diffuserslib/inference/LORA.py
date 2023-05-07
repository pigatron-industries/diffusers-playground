from safetensors.torch import load_file
import torch
import os

LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'

class LORAUse:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __hash__(self):
        return hash(self.name + str(self.weight))

    def __eq__(self,other):
        return self.name == other.name and self.weight== other.weight


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
        # TODO write custom lora loading function to take weight into account
        lora = torch.load(self.path)
        pipeline.load_lora_weights(lora)


class StableDiffusionLORA(LORA):
    def __init__(self, name, path):
        super().__init__(name, path)

    def add_to_model(self, pipeline, weight = 1, device="cuda"):
        state_dict = load_file(self.path, device=device)
        visited = []
        for key in state_dict:

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
            if len(state_dict[pair_keys[0]].shape) == 4:
                weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                curr_layer.weight.data += weight * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = state_dict[pair_keys[0]].to(torch.float32)
                weight_down = state_dict[pair_keys[1]].to(torch.float32)
                curr_layer.weight.data += weight * torch.mm(weight_up, weight_down)

            # update visited list
            for item in pair_keys:
                visited.append(item)