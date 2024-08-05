from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters
from dataclasses import dataclass
from diffusers import StableVideoDiffusionPipeline
import torch


@dataclass
class StableVideoDiffusionGenerationParameters(GenerationParameters):
    fps:int = 7


class StableVideoDiffusionPipelineWrapper(StableDiffusionPipelineWrapper):

    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        print(f"creating pipeline {cls.__name__}")
        super().__init__(cls=cls, params=params, device=device, **kwargs)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        # fp16 producing black images on mac
        pipeline_params['torch_dtype'] = torch.float16
        pipeline_params['variant'] = 'fp16'
        return pipeline_params
                

    def diffusers_inference(self, image, seed, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        output = self.pipeline(image, generator=generator, **kwargs)
        return output, seed


class StableVideoDiffusionGeneratePipelineWrapper(StableVideoDiffusionPipelineWrapper):

    def __init__(self, params:GenerationParameters, device):
        super().__init__(params=params, device=device, cls=StableVideoDiffusionPipeline)

    def inference(self, params:StableVideoDiffusionGenerationParameters):
        diffusers_params = {}
        initimageparams = params.getInitImage()
        if(initimageparams is None or initimageparams.image is None):
            raise ValueError("No init image specified")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height
        diffusers_params['seed'] = params.seed
        diffusers_params['num_frames'] = params.frames
        diffusers_params['fps'] = params.fps
        diffusers_params['min_guidance_scale'] = 1.0
        diffusers_params['max_guidance_scale'] = 3.0
        diffusers_params['motion_bucket_id'] = 127
        diffusers_params['noise_aug_strength'] = 0.02
        diffusers_params['decode_chunk_size'] = 8
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.frames[0], seed
