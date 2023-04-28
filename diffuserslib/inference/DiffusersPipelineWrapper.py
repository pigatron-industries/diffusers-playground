from ..DiffusersModelPresets import DiffusersModel
from ..StringUtils import mergeDicts
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, 
                        StableDiffusionImageVariationPipeline, StableDiffusionInstructPix2PixPipeline,
                        ControlNetModel, StableDiffusionControlNetPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)
import torch
import random
import sys

MAX_SEED = 4294967295

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersPipelineWrapper:
    def __init__(self, preset:DiffusersModel, controlmodel:str = None):
        self.preset = preset
        self.controlmodel = controlmodel

    def inference(self, **kwargs):
        pass


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, controlmodel=None, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        self.inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset, controlmodel)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()
        # pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)

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
        # if(self.cache_dir is not None):
        #     args['cache_dir'] = self.cache_dir
        return mergeDicts(args, kwargs)
    
    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed
    
    def loadScheduler(self, schedulerClass):
        if (isinstance(schedulerClass, str)):
            schedulerClass = str_to_class(schedulerClass)
        self.pipeline.scheduler = schedulerClass.from_config(self.pipeline.scheduler.config)
    
    def inference(self, prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)
        if(self.preset.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.pipeline(prompt, generator=generator, **kwargs).images[0]
        else:
            image = self.pipeline(prompt, generator=generator, **kwargs).images[0]
        return image, seed
    

class StableDiffusionTextToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionPipeline, preset, device, custom_pipeline = 'lpw_stable_diffusion', **kwargs)

    def inference(self, prompt, seed, **kwargs):
        return super().inference(prompt=prompt, seed=seed, **kwargs)


class StableDiffusionImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, device, **kwargs)

    def inference(self, prompt, seed, initimage, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, initimage=initimage, **kwargs)
    

class StableDiffusionInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, device, **kwargs)

    def inference(self, prompt, seed, initimage, maskimage, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, initimage=initimage, maskimage=maskimage, **kwargs)
    

class StableDiffusionControlNetPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, cls=DiffusionPipeline, **kwargs):
        controlnet = self.createControlNets(controlmodel)
        super().__init__(preset=preset, device=device, cls=cls, controlnet=controlnet, **kwargs)

    def createControlNets(self, controlmodel):
        self.controlmodel = controlmodel
        if(isinstance(controlmodel, list)):
            controlnet = []
            for cmodel in controlmodel:
                controlnet.append(ControlNetModel.from_pretrained(cmodel))
        else:
            controlnet = ControlNetModel.from_pretrained(controlmodel)
        return controlnet

    def inference(self, prompt, seed, **kwargs):
        return super().inference(prompt=prompt, seed=seed, **kwargs)
    

class StableDiffusionTextToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(cls=StableDiffusionControlNetPipeline, preset=preset, device=device, controlmodel=controlmodel, **kwargs)

    def inference(self, prompt, seed, **kwargs):
        return super().inference(prompt=prompt, seed=seed, **kwargs)
    

class StableDiffusionImageToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_img2img", preset=preset, device=device, controlmodel=controlmodel, **kwargs)

    def inference(self, prompt, seed, initimage, controlimage, **kwargs):
        initimage = initimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, initimage=initimage, controlnet_conditioning_image=controlimage, **kwargs)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_inpaint", preset=preset, device=device, controlmodel=controlmodel, **kwargs)

    def inference(self, prompt, seed, initimage, maskimage, controlimage, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, initimage=initimage, maskimage=maskimage, controlnet_conditioning_image=controlimage, **kwargs)