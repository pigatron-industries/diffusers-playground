from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from .GenerationParameters import GenerationParameters, InpainGenerationParameters, TileGenerationParameters
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Callable
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
import torch
import sys
import numpy as np
from compel import Compel


INPAINT_CONTROL_MODEL = "lllyasviel/control_v11p_sd15_inpaint"


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset, inferencedevice)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        # Allow custom pipeline to be specified by name
        # if (isinstance(cls, str)):
        #     args['custom_pipeline'] = cls
        #     cls = DiffusionPipeline
        if (preset.modelpath.endswith('.safetensors') or preset.modelpath.endswith('.ckpt')):
            self.pipeline = cls.from_single_file(preset.modelpath, load_safety_checker=self.safety_checker, **args).to(self.device)
        else:
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
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        # self.custom_pipeline = 'composable_stable_diffusion'
        # self.custom_pipeline = 'lpw_stable_diffusion'
        self.custom_pipeline = StableDiffusionPipeline
        super().__init__(self.custom_pipeline, preset, device, safety_checker=safety_checker)

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
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 strength=params.strength, scheduler=params.scheduler)


class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(DiffusionPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionControlNetPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, cls=DiffusionPipeline, safety_checker=True, **kwargs):
        controlnet = self.createControlNets(controlmodel)
        super().__init__(preset=preset, device=device, cls=cls, controlnet=controlnet, safety_checker=safety_checker)

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
    def __init__(self, preset:DiffusersModel, device, controlmodel=[], safety_checker=True, **kwargs):
        super().__init__(cls=StableDiffusionControlNetPipeline, preset=preset, device=device, controlmodel=controlmodel, safety_checker=safety_checker)

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
    def __init__(self, preset:DiffusersModel, device, controlmodel=[], safety_checker=True, **kwargs):
        super().__init__(cls=StableDiffusionControlNetImg2ImgPipeline, preset=preset, device=device, controlmodel=controlmodel, safety_checker=safety_checker)

    def inference(self, params:GenerationParameters):
        controlimages = []
        controlnet_conditioning_scale = []
        initimage = None
        for controlimageparams in params.controlimages:
            if(controlimageparams.model is None or controlimageparams.model == "initimage"):
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
#     def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
#         super().__init__(StableDiffusionInpaintPipeline, preset, device, safety_checker=safety_checker)

#     def inference(self, prompt, negprompt, seed, initimage, maskimage, scale, steps, scheduler, strength=1.0, **kwargs):
#         initimage = initimage.convert("RGB")
#         maskimage = maskimage.convert("RGB")
#         return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, 
#                                  strength=strength, scheduler=scheduler, width=initimage.width, height=initimage.height)


class StableDiffusionInpaintPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, preset=preset, device=device, controlmodel=INPAINT_CONTROL_MODEL, safety_checker=safety_checker)

    def inference(self, params:GenerationParameters):
        initimage = None
        maskimage = None
        for controlimageparams in params.controlimages:
            if(controlimageparams.model is None or controlimageparams.model == "initimage"):
                initimage = controlimageparams.image.convert("RGB")
            elif(controlimageparams.model == "maskimage"):
                maskimage = controlimageparams.image.convert("RGB")
        if(initimage is None or maskimage is None):
            raise ValueError("Must provide both initimage and maskimage")
        inpaint_pt = make_inpaint_condition(initimage=initimage, maskimage=maskimage)
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, control_image=inpaint_pt, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, strength=params.strength, scheduler=params.scheduler, width=initimage.width, height=initimage.height)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel=[], safety_checker=True, **kwargs):
        if(not isinstance(controlmodel, list)):
            controlmodel = [controlmodel]
        controlmodel.append(INPAINT_CONTROL_MODEL)
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, preset=preset, device=device, controlmodel=controlmodel, safety_checker=safety_checker)


    def inference(self, params:GenerationParameters):
        initimage = None
        maskimage = None
        controlnet_conditioning_scale = []
        controlimages = []
        for controlimageparams in params.controlimages:
            if(controlimageparams.model is None or controlimageparams.model == "initimage"):
                initimage = controlimageparams.image.convert("RGB")
            elif(controlimageparams.model == "maskimage"):
                maskimage = controlimageparams.image.convert("RGB")
            else:
                controlimages.append(pil_to_pt(controlimageparams.image.convert("RGB")))
                controlnet_conditioning_scale.append(controlimageparams.condscale)
        if(initimage is None or maskimage is None):
            raise ValueError("Must provide both initimage and maskimage")
        inpaint_pt = make_inpaint_condition(initimage=initimage, maskimage=maskimage)
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