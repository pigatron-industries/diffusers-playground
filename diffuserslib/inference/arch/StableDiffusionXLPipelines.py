from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                        StableDiffusionXLControlNetPipeline, ControlNetModel,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler)
from compel import Compel, ReturnedEmbeddingsType
import torch


class NoWatermark:
    def apply_watermark(self, img):
        return img


class StableDiffusionXLPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, cls, preset:DiffusersModel, params:GenerationParameters, device, **kwargs):
        super().__init__(cls=cls, preset=preset, params=params, device=device, **kwargs)
        self.pipeline.watermark = NoWatermark()

    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        if "|" in prompt:
            prompt1, prompt2 = prompt.split("|")
        else:
            prompt1 = prompt
            prompt2 = prompt

        if "|" in negative_prompt:
            negative_prompt1, negative_prompt2 = negative_prompt.split("|")
        else:
            negative_prompt1 = negative_prompt
            negative_prompt2 = negative_prompt

        compel1 = Compel(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=False,
        )

        compel2 = Compel(
            tokenizer=self.pipeline.tokenizer_2,
            text_encoder=self.pipeline.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
        )

        conditioning1 = compel1(prompt1)
        conditioning2, pooled = compel2(prompt2)
        conditioning = torch.cat((conditioning1, conditioning2), dim=-1)

        negative_conditioning1 = compel1(negative_prompt1)
        negative_conditioning2, negative_pooled = compel2(negative_prompt2)
        negative_conditioning = torch.cat((negative_conditioning1, negative_conditioning2), dim=-1)

        image = self.pipeline(prompt_embeds=conditioning,
                              pooled_prompt_embeds=pooled,
                              negative_prompt_embeds=negative_conditioning,
                              negative_pooled_prompt_embeds=negative_pooled,
                              generator=generator, **kwargs).images[0]
        return image, seed

    def add_embeddings(self, token, embeddings):
        self.add_embedding_to_text_encoder(token, embeddings[0], self.pipeline.tokenizer, self.pipeline.text_encoder)
        self.add_embedding_to_text_encoder(token, embeddings[1], self.pipeline.tokenizer_2, self.pipeline.text_encoder_2)


class StableDiffusionXLTextToImagePipelineWrapper(StableDiffusionXLPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionXLPipeline, preset=preset, params=params, device=device)

    def inference(self, params:GenerationParameters):
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, width=params.width, height=params.height, seed=params.seed, 
                                           guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionXLImageToImagePipelineWrapper(StableDiffusionXLPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionXLImg2ImgPipeline, preset=preset, params=params, device=device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                           strength=params.strength, scheduler=params.scheduler)
    

class StableDiffusionXLInpaintPipelineWrapper(StableDiffusionXLPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionXLInpaintPipeline, preset=preset, params=params, device=device)

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


class StableDiffusionXLControlNetPipelineWrapper(StableDiffusionXLPipelineWrapper):
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
        

class StableDiffusionXLTextToImageControlNetPipelineWrapper(StableDiffusionXLControlNetPipelineWrapper):
    def __init__(self, preset:DiffusersModel, params:GenerationParameters, device):
        super().__init__(cls=StableDiffusionXLControlNetPipeline, preset=preset, params=params, device=device)

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
    