from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        StableVideoDiffusionPipeline)
from compel import Compel, ReturnedEmbeddingsType
import torch


class StableVideoDiffusionPipelineWrapper(StableDiffusionPipelineWrapper):

    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        print(f"creating pipeline {cls.__name__}")
        super().__init__(cls=cls, params=params, device=device, **kwargs)
                

    def diffusers_inference(self, image, seed, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        output = self.pipeline(image, decode_chunk_size = 8, generator=generator, **kwargs)
        return output, seed


class StableVideoDiffusionGeneratePipelineWrapper(StableVideoDiffusionPipelineWrapper):

    def __init__(self, params:GenerationParameters, device):
        super().__init__(params=params, device=device, cls=StableVideoDiffusionPipeline)

    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        initimageparams = params.getInitImage()
        if(initimageparams is None or initimageparams.image is None):
            raise ValueError("No init image specified")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['seed'] = params.seed
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.frames[0], seed
