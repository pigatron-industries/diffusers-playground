from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import KandinskyPriorPipeline, KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyInpaintPipeline
import torch


class KandinskyPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset, params, inferencedevice)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        self.pipeline_prior = KandinskyPriorPipeline.from_pretrained(preset.data['id2'], **args).to(self.device)
        self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)

    def createPipelineArgs(self, preset, **kwargs):
        args = {}
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['variant'] = preset.revision
            if(preset.revision == 'fp16'):
                args['torch_dtype'] = torch.float16
        return mergeDicts(args, kwargs)
    
    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, **kwargs):
        generator, seed = self.createGenerator(seed)
        image_embeds, negimage_embeds = self.pipeline_prior(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=False)
        image = self.pipeline(prompt=prompt, image_embeds=image_embeds, negative_image_embeds=negimage_embeds, guidance_scale=guidance_scale, **kwargs).images[0]
        return image, seed


class KandinskyTextToImagePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(KandinskyPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, width=params.width, height=params.height, seed=params.seed, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler)


class KandinskyImageToImagePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(KandinskyImg2ImgPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 strength=params.strength, scheduler=params.scheduler)


class KandinskyInpaintPipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(KandinskyInpaintPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None):
            raise ValueError("Must provide both initimage and maskimage")
        initimage = initimageparams.image.convert("RGB")
        maskimage = maskimageparams.image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, 
                                           guidance_scale=params.cfgscale, num_inference_steps=params.steps, strength=params.strength, scheduler=params.scheduler, 
                                           width=initimage.width, height=initimage.height)


class KandinskyInterpolatePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(KandinskyPipeline, preset, params, device)

    def inference(self, params:GenerationParameters):
        generator, seed = self.createGenerator(params.seed)
        prior_interpolate = self.pipeline_prior.interpolate(params.prompt, params.promptweights)
        image = self.pipeline("", **prior_interpolate, guidance_scale=params.cfgscale, num_inference_steps=params.steps, width=params.width, height=params.height).images[0]
        return image, seed