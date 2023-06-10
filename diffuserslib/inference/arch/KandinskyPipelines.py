from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import KandinskyPriorPipeline, KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyInpaintPipeline
import torch


class KandinskyPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        self.safety_checker = safety_checker
        self.device = device
        self.inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(preset, cls, **kwargs)
        super().__init__(preset)

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
    
    def inference(self, prompt, negprompt, seed, guidance_scale=4.0, **kwargs):
        generator, seed = self.createGenerator(seed)
        image_embeds, negimage_embeds = self.pipeline_prior(prompt=prompt, negative_prompt=negprompt, generator=generator, guidance_scale=guidance_scale, return_dict=False)
        image = self.pipeline(prompt=prompt, image_embeds=image_embeds, negative_image_embeds=negimage_embeds, guidance_scale=guidance_scale, **kwargs).images[0]
        return image, seed


class KandinskyTextToImagePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(KandinskyPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, seed, scale, steps, width, height, **kwargs):
        return super().inference(prompt=prompt, negprompt=negprompt, seed=seed, guidance_scale=scale, num_inference_steps=steps, width=width, height=height)


class KandinskyImageToImagePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(KandinskyImg2ImgPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, initimage, seed, scale, steps, strength, width, height, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, negprompt=negprompt, image=initimage, seed=seed, guidance_scale=scale, num_inference_steps=steps, strength=strength, 
                                 width=initimage.width, height=initimage.height)


class KandinskyInpaintPipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(KandinskyInpaintPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, initimage, maskimage, seed, scale, steps, width, height, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return super().inference(prompt=prompt, negprompt=negprompt, image=initimage, mask_image=maskimage, seed=seed, guidance_scale=scale, num_inference_steps=steps,
                                 width=initimage.width, height=initimage.height)


class KandinskyInterpolatePipelineWrapper(KandinskyPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(KandinskyPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, images_prompts, weights, seed, scale, steps, width, height, **kwargs):
        generator, seed = self.createGenerator(seed)
        prior_interpolate = self.pipeline_prior.interpolate(images_prompts, weights)
        image = self.pipeline("", **prior_interpolate, guidance_scale=scale, num_inference_steps=steps, width=width, height=height, **kwargs).images[0]
        return image, seed