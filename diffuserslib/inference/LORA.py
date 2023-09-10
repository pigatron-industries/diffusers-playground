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
        return cls(name, path)
        
    def add_to_model(self, pipeline, weight = 1, device="cuda"):
        pipeline.load_lora_weights(self.path)
        pipeline.fuse_lora(lora_scale = weight)
