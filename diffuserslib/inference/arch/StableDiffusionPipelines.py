from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
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
import sys


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        self.inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset)

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
        return mergeDicts(args, kwargs)
    
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
        super().__init__(StableDiffusionPipeline, preset, device) #custom_pipeline = 'lpw_stable_diffusion',

    def inference(self, prompt, negprompt, seed, scale, steps, scheduler, **kwargs):
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)


class StableDiffusionImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, device)

    def inference(self, prompt, negprompt, seed, initimage, scale, scheduler, strength, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, guidance_scale=scale, strength=strength, scheduler=scheduler)
    

class StableDiffusionInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionInpaintPipeline, preset, device)

    def inference(self, prompt, negprompt, seed, initimage, maskimage, scale, steps, scheduler, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)
    

class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(DiffusionPipeline, preset, device)

    def inference(self, prompt, seed, initimage, scale, steps, scheduler, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)


class StableDiffusionControlNetPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, cls=DiffusionPipeline, **kwargs):
        controlnet = self.createControlNets(controlmodel)
        super().__init__(preset=preset, device=device, cls=cls, controlnet=controlnet)

    def createControlNets(self, controlmodel):
        self.controlmodel = controlmodel
        if(isinstance(controlmodel, list)):
            controlnet = []
            for cmodel in controlmodel:
                controlnet.append(ControlNetModel.from_pretrained(cmodel))
        else:
            controlnet = ControlNetModel.from_pretrained(controlmodel)
        return controlnet
    

class StableDiffusionTextToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(cls=StableDiffusionControlNetPipeline, preset=preset, device=device, controlmodel=controlmodel)

    def inference(self, prompt, negprompt, seed, scale, steps, scheduler, **kwargs):
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)
    

class StableDiffusionImageToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_img2img", preset=preset, device=device, controlmodel=controlmodel)

    def inference(self, prompt, negprompt, seed, initimage, controlimage, scale, strength, scheduler, **kwargs):
        initimage = initimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, controlnet_conditioning_image=controlimage, 
                                 guidance_scale=scale, strength=strength, scheduler=scheduler)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_inpaint", preset=preset, device=device, controlmodel=controlmodel)

    def inference(self, prompt, negprompt, seed, initimage, maskimage, controlimage, scale, steps, scheduler, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, 
                                 controlnet_conditioning_image=controlimage, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)