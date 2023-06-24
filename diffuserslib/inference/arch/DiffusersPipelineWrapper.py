from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Tuple
from PIL import Image
import sys
import random
import torch


MAX_SEED = 4294967295


class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel):
        self.preset = preset

    def inference(self, **kwargs) -> Tuple[Image.Image, int]: # type: ignore
        pass

    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed
    
    def isEqual(self, cls, modelid, **kwargs):
        print("isEqual", cls, modelid, self.preset.modelid)
        return cls == self.__class__ and self.preset.modelid == modelid
