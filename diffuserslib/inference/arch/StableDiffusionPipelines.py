from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageParameters, IMAGETYPE_MASKIMAGE, IMAGETYPE_INITIMAGE, IMAGETYPE_CONTROLIMAGE
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Callable, List
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, 
                        StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetPipeline,
                        StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)
from transformers import CLIPTextModel
import torch
import sys
import numpy as np
from compel import Compel


INPAINT_CONTROL_MODEL = "lllyasviel/control_v11p_sd15_inpaint"


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset, params, inferencedevice)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        if (preset.modelpath.endswith('.safetensors') or preset.modelpath.endswith('.ckpt')):
            self.pipeline = cls.from_single_file(preset.modelpath, load_safety_checker=self.safety_checker, **args).to(self.device)
        else:
            # CLIP skip implementation, but breaks lora loading
            # text_encoder = CLIPTextModel.from_pretrained(preset.modelpath, subfolder="text_encoder", num_hidden_layers=11)
            # self.pipeline = cls.from_pretrained(preset.modelpath, text_encoder=text_encoder, **args).to(self.device)
            self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)
            
        self.pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()
        # pipeline.enable_xformers_memory_efficient_attention()

    def createPipelineArgs(self, preset, **kwargs):
        args = {}
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['revision'] = preset.revision
            if(preset.revision == 'fp16'):
                args['torch_dtype'] = torch.float16
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        return mergeDicts(args, kwargs)
    
    def loadScheduler(self, schedulerClass):
        if (isinstance(schedulerClass, str)):
            schedulerClass = str_to_class(schedulerClass)
        self.pipeline.scheduler = schedulerClass.from_config(self.pipeline.scheduler.config)
    
    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        compel = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)
        conditioning = compel(prompt)
        negative_conditioning = compel(negative_prompt)
        # [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

        if(self.preset.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs).images[0]
        else:
            image = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs).images[0]
        return image, seed
    

class StableDiffusionTextToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        # self.custom_pipeline = 'composable_stable_diffusion'
        # self.custom_pipeline = 'lpw_stable_diffusion'
        self.custom_pipeline = StableDiffusionPipeline
        super().__init__(self.custom_pipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        # args = {}
        # if (self.custom_pipeline == 'composable_stable_diffusion'):
        #     prompts = prompt.split("|")
        #     weights = None
        #     if (len(prompts) > 1):
        #         negprompt = [negprompt] * len(prompts)
        #         weights = []
        #         for promptpart in prompts:
        #             weight = promptpart.split(" ")[-1]
        #             if (weight.isnumeric()):
        #                 weights.append(weight)
        #             else:
        #                 weights.append("1")
        #         weights = " | ".join(weights)
        #     args['weights'] = weights

        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, width=params.width, height=params.height, seed=params.seed, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 strength=params.strength, scheduler=params.scheduler)


class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(DiffusionPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionControlNetPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device, cls=DiffusionPipeline, extracontrolmodels:List[str]=[]):
        controlmodel = []
        for extracontrolmodel in extracontrolmodels:
            controlmodel.append(extracontrolmodel)
        for controlimageparams in params.getControlImages():
            controlmodel.append(controlimageparams.model)
        controlnet = self.createControlNets(controlmodel)
        super().__init__(preset=preset, params=params, device=device, cls=cls, controlnet=controlnet)

    def createControlNets(self, controlmodel):
        self.controlmodel = controlmodel
        if(isinstance(controlmodel, list)):
            controlnet = []
            for cmodel in controlmodel:
                controlnet.append(ControlNetModel.from_pretrained(cmodel))
            if(len(controlnet) == 1):
                controlnet = controlnet[0]
        else:
            controlnet = ControlNetModel.from_pretrained(controlmodel)
        return controlnet
    
    def isEqual(self, cls, modelid, controlmodel=None, **kwargs):
        if(controlmodel is None):
            return super().isEqual(cls, modelid)
        else:
            return super().isEqual(cls, modelid) and self.controlmodel == controlmodel
    

class StableDiffusionTextToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionControlNetPipeline, preset=preset, params=params, device=device)

    def inference(self, params:GenerationParameters):
        if(len(params.controlimages) == 1):
            controlnet_conditioning_scale = params.controlimages[0].condscale
        else:
            controlnet_conditioning_scale = []
            for controlimage in params.controlimages:
                controlnet_conditioning_scale.append(controlimage.condscale)
        controlimages = []
        for controlimage in params.controlimages:
            controlimages.append(controlimage.image.convert("RGB"))
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=controlimages, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler, controlnet_conditioning_scale=controlnet_conditioning_scale)
    

class StableDiffusionImageToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, params:GenerationParameters):
        super().__init__(cls=StableDiffusionControlNetImg2ImgPipeline, preset=preset, params=params, device=device)

    def inference(self, params:GenerationParameters):
        controlimages = []
        controlnet_conditioning_scale = []
        initimage = None
        for controlimageparams in params.controlimages:
            if(controlimageparams.model is None or controlimageparams.type == IMAGETYPE_INITIMAGE):
                initimage = controlimageparams.image.convert("RGB")
            else:
                controlimages.append(controlimageparams.image.convert("RGB"))
                controlnet_conditioning_scale.append(controlimageparams.condscale)
        if(len(controlnet_conditioning_scale) == 1):
            controlnet_conditioning_scale = controlnet_conditioning_scale[0]
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, control_image=controlimages, 
                                 guidance_scale=params.cfgscale, strength=params.strength, scheduler=params.scheduler, controlnet_conditioning_scale=controlnet_conditioning_scale)
    

# Standard stable diffusion inpaint pipeline
# class StableDiffusionInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
#     def __init__(self, preset:DiffusersModel, device):
#         super().__init__(StableDiffusionInpaintPipeline, preset, device)

#     def inference(self, prompt, negprompt, seed, initimage, maskimage, scale, steps, scheduler, strength=1.0, **kwargs):
#         initimage = initimage.convert("RGB")
#         maskimage = maskimage.convert("RGB")
#         return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, 
#                                  strength=strength, scheduler=scheduler, width=initimage.width, height=initimage.height)


class StableDiffusionInpaintPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, preset=preset, params=params, device=device, extracontrolmodels = [INPAINT_CONTROL_MODEL])

    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None):
            raise ValueError("Must provide both initimage and maskimage")
        initimage = initimageparams.image.convert("RGB")
        maskimage = maskimageparams.image.convert("RGB")
        inpaint_pt = make_inpaint_condition(initimage=initimage, maskimage=maskimage)
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, control_image=inpaint_pt, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, strength=params.strength, scheduler=params.scheduler, width=initimage.width, height=initimage.height)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, preset=preset, params=params, device=device, extracontrolmodels = [INPAINT_CONTROL_MODEL])


    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None):
            raise ValueError("Must provide both initimage and maskimage")
        initimage = initimageparams.image.convert("RGB")
        maskimage = maskimageparams.image.convert("RGB")

        controlnet_conditioning_scale = [1.0]
        controlimages = [make_inpaint_condition(initimage=initimage, maskimage=maskimage)]
        for controlimageparams in params.controlimages:
            controlimages.append(pil_to_pt(controlimageparams.image))
            controlnet_conditioning_scale.append(controlimageparams.condscale)

        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, 
                                 control_image=controlimages, guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler, 
                                 width=initimage.width, height=initimage.height, controlnet_conditioning_scale=controlnet_conditioning_scale)
    

def make_inpaint_condition(initimage:Image.Image, maskimage:Image.Image) -> torch.Tensor:
    npinitimage = np.array(initimage.convert("RGB")).astype(np.float32) / 255.0
    npmaskimage = np.array(maskimage.convert("L")).astype(np.float32) / 255.0
    npinitimage[npmaskimage > 0.5] = -1.0  # set as masked pixel
    npinitimage = np.expand_dims(npinitimage, 0).transpose(0, 3, 1, 2)
    npinitimage = torch.from_numpy(npinitimage)
    return npinitimage


def pil_to_pt(image:Image.Image) -> torch.Tensor:
    npimage = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    npimage = np.expand_dims(npimage, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(npimage)