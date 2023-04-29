from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, 
                        StableDiffusionImageVariationPipeline, StableDiffusionInstructPix2PixPipeline,
                        ControlNetModel, StableDiffusionControlNetPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)
import torch
import random
import sys

MAX_SEED = 4294967295

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, controlmodel:str = None):
        self.preset = preset
        self.controlmodel = controlmodel

    def inference(self, **kwargs):
        pass
