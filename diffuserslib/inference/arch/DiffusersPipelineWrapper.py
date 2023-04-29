from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
import sys


class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, controlmodel:str = None):
        self.preset = preset
        self.controlmodel = controlmodel

    def inference(self, **kwargs):
        pass
