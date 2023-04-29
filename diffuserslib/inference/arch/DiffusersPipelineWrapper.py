from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
import sys


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, controlmodel:str = None):
        self.preset = preset
        self.controlmodel = controlmodel

    def inference(self, **kwargs):
        pass
