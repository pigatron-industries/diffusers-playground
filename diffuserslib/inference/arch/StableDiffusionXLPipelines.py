from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from PIL import Image
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)



class StableDiffusionXLTextToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(StableDiffusionXLPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, width, height, seed, scale, steps, scheduler, **kwargs):
        return super().inference(prompt=prompt, negative_prompt=negprompt, width=width, height=height, seed=seed, guidance_scale=scale, num_inference_steps=steps, scheduler=scheduler)


class StableDiffusionXLImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(StableDiffusionXLImg2ImgPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, seed, initimage, scale, scheduler, strength, **kwargs):
        initimage = initimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, guidance_scale=scale, strength=strength, scheduler=scheduler)

class StableDiffusionXLInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(StableDiffusionXLInpaintPipeline, preset, device, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, seed, initimage, maskimage, scale, steps, scheduler, strength=1.0, **kwargs):
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, 
                                 strength=strength, scheduler=scheduler, width=initimage.width, height=initimage.height)
