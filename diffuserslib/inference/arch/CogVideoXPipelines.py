from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, VideoGenerationParameters
from dataclasses import dataclass
import torch
from diffuserslib.inference.GenerationParameters import ControlImageType
from PIL import Image


class CogVideoXPipelineWrapper(DiffusersPipelineWrapper):

    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        print(f"creating pipeline {cls.__name__}")
        super().__init__(cls=cls, params=params, device=device, **kwargs)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        pipeline_params['torch_dtype'] = torch.float16
        return pipeline_params
                

    def diffusers_inference(self, image, seed, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        # if(scheduler is not None):
        #     self.loadScheduler(scheduler)
        output = self.pipeline(image, generator=generator, **kwargs)
        return output, seed


class CogVideoXGeneratePipelineWrapper(CogVideoXPipelineWrapper):

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXVideoToVideoPipeline
        initimageparams = params.getImage(ControlImageType.IMAGETYPE_INITIMAGE)
        if(initimageparams is None):
            return CogVideoXPipeline
        elif(isinstance(initimageparams.image, Image.Image)):
            return CogVideoXImageToVideoPipeline
        else:
            return CogVideoXVideoToVideoPipeline



    def inference(self, params:VideoGenerationParameters):
        diffusers_params = {}
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_frames'] = params.frames
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['seed'] = params.seed

        initimageparams = params.getInitImage()
        if(initimageparams is not None and isinstance(initimageparams.image, Image.Image)):
            diffusers_params['image'] = initimageparams.image.convert("RGB")

        output, seed = super().diffusers_inference(**diffusers_params)
        return output.frames[0], seed
