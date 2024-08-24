from .StableDiffusionPipelines import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        FluxPipeline, FluxControlNetPipeline,
                        # Conditioning models
                        FluxControlNetModel,
                        # Schedulers
                        FlowMatchEulerDiscreteScheduler)
import diffusers
import torch
from safetensors import safe_open
from diffuserslib.scripts.convert_flux_lora import convert_sd_scripts_to_ai_toolkit


class FluxPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.lora_names = []
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, controlnet_cls = FluxControlNetModel, **kwargs)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        if(self.features.controlnet):
            self.addPipelineParamsControlNet(params, pipeline_params)
        return pipeline_params
    
    def addPipelineParamsControlNet(self, params:GenerationParameters, pipeline_params):
        args = {}
        if(self.dtype is not None):
            args['torch_dtype'] = self.dtype
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        if(len(controlnetparams) > 1):
            raise ValueError("Only one controlnet model is supported by flux pipeline.")
        controlnet = self.controlnet_cls.from_pretrained(controlnetparams[0].model, **args)
        pipeline_params['controlnet'] = controlnet
        return pipeline_params
    
    def addInferenceParamsControlNet(self, params:GenerationParameters, diffusers_params):
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        if(controlnetparams[0].image is not None and controlnetparams[0].condscale > 0):
            diffusers_params['control_image'] = controlnetparams[0].image
            diffusers_params['controlnet_conditioning_scale'] = controlnetparams[0].condscale
    
    def diffusers_inference(self, prompt, seed, guidance_scale=4.0, scheduler=None, negative_prompt=None, clip_skip=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed
    

    def load_lora_weights(self, path:str):
        state_dict = {}
        sdscripts_format = False
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            for k in f.keys():
                if(k.startswith('lora_unet')):
                    sdscripts_format = True
                state_dict[k] = f.get_tensor(k)
        if(sdscripts_format):
            state_dict = convert_sd_scripts_to_ai_toolkit(state_dict)
        return state_dict
        

    def add_lora(self, lora):
        if(lora.name not in self.lora_names):
            state_dict = self.load_lora_weights(lora.path)
            self.lora_names.append(lora.name)
            self.pipeline.load_lora_weights(state_dict, adapter_name=lora.name.split('.', 1)[0])


    def add_loras(self, loras, weights:List[float]):
        for lora in loras:
            self.add_lora(lora)
        lora_weights = []
        lora_names = []
        for i, lora in enumerate(loras):
            lora_weights.append(weights[i])
            lora_names.append(lora.name.split('.', 1)[0])

        if(hasattr(self.pipeline, 'set_adapters')):
            self.pipeline.set_adapters(lora_names, lora_weights)


class FluxGeneratePipelineWrapper(FluxPipelineWrapper):

    PIPELINE_MAP = {
        #img2img,   inpaint, controlnet
        (False,     False,   False):    FluxPipeline,
        (False,     False,   True):     FluxControlNetPipeline
    }


    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint, self.features.controlnet)]
