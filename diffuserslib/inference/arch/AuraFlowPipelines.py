from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from diffusers.pipelines.aura_flow.pipeline_aura_flow import AuraFlowPipeline
import torch


class AuraFlowPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, **kwargs)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        pipeline_params['torch_dtype'] = torch.float16
        return pipeline_params
    
    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, scheduler=None, clip_skip=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed


class AuraFlowGeneratePipelineWrapper(AuraFlowPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    inpaint
        (False,     False):    AuraFlowPipeline
    }


    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]
