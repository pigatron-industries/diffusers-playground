from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import PixArtSigmaPipeline
import torch


class PixartSigmaPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params, cls, **kwargs)
        super().__init__(params, inferencedevice)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    
    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed


class PixartSigmaGeneratePipelineWrapper(PixartSigmaPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    inpaint
        (False,     False):    PixArtSigmaPipeline
    }


    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]
