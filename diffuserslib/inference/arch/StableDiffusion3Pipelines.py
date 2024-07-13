from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline,
                        # Conditioning models
                        T2IAdapter, ControlNetModel,
                        # Schedulers
                        FlowMatchEulerDiscreteScheduler)
import torch


class StableDiffusion3PipelineWrapper(StableDiffusionPipelineWrapper):

    def __init__(self, cls, params:GenerationParameters, device, dtype=None):
        super().__init__(cls=cls, params=params, device=device, dtype=dtype)
                

    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        # if(scheduler is not None):
        #     self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        output = self.pipeline(prompt = prompt,
                              negative_prompt = negative_prompt,
                              generator=generator, **kwargs)
        return output, seed



class StableDiffusion3GeneratePipelineWrapper(StableDiffusion3PipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    controlnet, t2iadapter, inpaint
        (False,     False,      False,      False):    StableDiffusion3Pipeline,
        # (False,     True,       False,      False):    StableDiffusion3ControlNetPipeline,
        # (False,     False,      True,       False):    StableDiffusion3AdapterPipeline,
        (True,      False,      False,      False):    StableDiffusion3Img2ImgPipeline,
        # (True,      True,       False,      False):    StableDiffusion3ControlNetImg2ImgPipeline,
        # (True,      False,      False,      True):     StableDiffusion3InpaintPipeline,
        # (True,      True,       False,      True):     StableDiffusion3ControlNetInpaintPipeline,
    }


    def __init__(self, params:GenerationParameters, device):
        self.dtype = None
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)

        super().__init__(params=params, device=device, cls=cls)

        if(self.features.ipadapter):
            self.initIpAdapter(params)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.controlnet, self.features.t2iadapter, self.features.inpaint)]  


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        if(self.features.controlnet):
            self.addPipelineParamsControlNet(params, pipeline_params)
        if(self.features.t2iadapter):
            self.addPipelineParamsT2IAdapter(params, pipeline_params)
        if(self.features.ipadapter):
            self.addPipelineParamsIpAdapter(params, pipeline_params)
        return pipeline_params

