from ..DiffusersModelPresets import DiffusersModel
from diffusers import DiffusionPipeline

class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, pipeline:DiffusionPipeline, controlmodel:str = None):
        self.preset = preset
        self.pipeline = pipeline
        self.controlmodel = controlmodel

