from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from PIL import Image
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                        StableDiffusionXLControlNetPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)
from compel import Compel, ReturnedEmbeddingsType


class StableDiffusionXLPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, device, safety_checker=True, **kwargs):
        super().__init__(cls, preset, device, safety_checker=safety_checker, **kwargs)

    def inference(self, prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        compel = Compel(tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2] , text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2], 
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
        conditioning, pooled = compel(prompt)

        image = self.pipeline(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, generator=generator, **kwargs).images[0]
        return image, seed


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


class StableDiffusionXLControlNetPipelineWrapper(StableDiffusionXLPipelineWrapper):
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
        

class StableDiffusionXLTextToImageControlNetPipelineWrapper(StableDiffusionXLControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, device, controlmodel=[], safety_checker=True, **kwargs):
        super().__init__(cls=StableDiffusionXLControlNetPipeline, preset=preset, device=device, controlmodel=controlmodel, safety_checker=safety_checker)

    def inference(self, prompt, negprompt, seed, scale, steps, scheduler, controlimage, controlnet_conditioning_scale, **kwargs):
        return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=controlimage, guidance_scale=scale, 
                                 num_inference_steps=steps, scheduler=scheduler, controlnet_conditioning_scale=controlnet_conditioning_scale)