from .StableDiffusionPipelines import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        FluxPipeline,
                        # Conditioning models
                        T2IAdapter, ControlNetModel,
                        # Schedulers
                        FlowMatchEulerDiscreteScheduler)
import diffusers
import torch


_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)

diffusers.models.transformers.transformer_flux.rope = new_flux_rope




class FluxPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, **kwargs)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    
    def diffusers_inference(self, prompt, seed, guidance_scale=4.0, scheduler=None, negative_prompt=None, clip_skip=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed


class FluxGeneratePipelineWrapper(FluxPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    inpaint
        (False,     False):    FluxPipeline
    }


    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]
