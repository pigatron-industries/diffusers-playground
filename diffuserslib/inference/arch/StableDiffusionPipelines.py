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
        super().__init__(StableDiffusionPipeline, preset, device,  **kwargs) #custom_pipeline = 'lpw_stable_diffusion',

    def inference(self, prompt, seed, scale, steps, **kwargs):
        return super().inference(prompt=prompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, **kwargs)


class StableDiffusionImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionImg2ImgPipeline, preset, device, **kwargs)

    def inference(self, prompt, seed, initimage, scale, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, guidance_scale=scale, **kwargs)
    

class StableDiffusionInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(StableDiffusionInpaintPipeline, preset, device, **kwargs)

    def inference(self, prompt, seed, initimage, maskimage, scale, steps, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, **kwargs)
    

class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, **kwargs):
        super().__init__(DiffusionPipeline, preset, device, **kwargs)

    def inference(self, prompt, seed, initimage, scale, steps, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, guidance_scale=scale, num_inference_steps=steps, **kwargs)


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

    def inference(self, prompt, seed, scale, steps, **kwargs):
        return super().inference(prompt=prompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, **kwargs)
    

class StableDiffusionImageToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_img2img", preset=preset, device=device, controlmodel=controlmodel, **kwargs)

    def inference(self, prompt, seed, initimage, controlimage, scale, **kwargs):
        initimage = initimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, controlnet_conditioning_image=controlimage, guidance_scale=scale, **kwargs)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel, **kwargs):
        super().__init__(custom_pipeline="stable_diffusion_controlnet_inpaint", preset=preset, device=device, controlmodel=controlmodel, **kwargs)

    def inference(self, prompt, seed, initimage, maskimage, controlimage, scale, steps, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        if(isinstance(controlimage, list)):
            controlimage = list(map(lambda x: x.convert("RGB"), controlimage))
        else:
            controlimage = controlimage.convert("RGB")
        return super().inference(prompt=prompt, seed=seed, image=initimage, mask_image=maskimage, controlnet_conditioning_image=controlimage, guidance_scale=scale, num_inference_steps=steps, **kwargs)