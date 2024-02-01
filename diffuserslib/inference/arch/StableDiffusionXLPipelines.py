from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                        StableDiffusionXLControlNetPipeline, StableDiffusionXLAdapterPipeline, StableDiffusionXLControlNetImg2ImgPipeline,
                        StableDiffusionXLControlNetInpaintPipeline,
                        # Conditioning models
                        T2IAdapter, ControlNetModel,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler, LCMScheduler)
from compel import Compel, ReturnedEmbeddingsType
import torch


class NoWatermark:
    def apply_watermark(self, img):
        return img


class StableDiffusionXLPipelineWrapper(StableDiffusionPipelineWrapper):
    LCM_LORA_MODEL = "latent-consistency/lcm-lora-sdxl"

    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        super().__init__(cls=cls, params=params, device=device, **kwargs)
        self.pipeline.watermark = NoWatermark()
        if(params.scheduler == "LCMScheduler"):
            self.pipeline.load_lora_weights(self.LCM_LORA_MODEL)
            self.pipeline.fuse_lora()
                

    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        if(self.params.scheduler == "LCMScheduler"):
            kwargs['guidance_scale'] = 0

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

        output = self.pipeline(prompt_embeds=conditioning,
                              pooled_prompt_embeds=pooled,
                              negative_prompt_embeds=negative_conditioning,
                              negative_pooled_prompt_embeds=negative_pooled,
                              generator=generator, **kwargs)
        return output, seed


    def add_embeddings(self, token, embeddings):
        self.add_embedding_to_text_encoder(token, embeddings[0], self.pipeline.tokenizer, self.pipeline.text_encoder)
        self.add_embedding_to_text_encoder(token, embeddings[1], self.pipeline.tokenizer_2, self.pipeline.text_encoder_2)



class StableDiffusionXLGeneratePipelineWrapper(StableDiffusionXLPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    controlnet, t2iadapter, inpaint
        (False,     False,      False,      False):    StableDiffusionXLPipeline,
        (False,     True,       False,      False):    StableDiffusionXLControlNetPipeline,
        (False,     False,      True,       False):    StableDiffusionXLAdapterPipeline,
        (True,      False,      False,      False):    StableDiffusionXLImg2ImgPipeline,
        (True,      True,       False,      False):    StableDiffusionXLControlNetImg2ImgPipeline,
        (True,      False,      False,      True):     StableDiffusionXLInpaintPipeline,
        (True,      True,       False,      True):     StableDiffusionXLControlNetInpaintPipeline,
    }


    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        if(not self.features.controlnet and not self.features.t2iadapter):
            super().__init__(params=params, device=device, cls=cls)
        else:
            controlmodelids = self.getConditioningModels(params)
            if(self.features.t2iadapter):
                conditioningmodels = self.createConditioningModels(controlmodelids, T2IAdapter)
                super().__init__(params=params, device=device, cls=cls, adapter=conditioningmodels)
            elif(self.features.controlnet):
                conditioningmodels = self.createConditioningModels(controlmodelids, ControlNetModel)
                super().__init__(params=params, device=device, cls=cls, controlnet=conditioningmodels)
        if(self.features.ipadapter):
            self.initIpAdapter(params)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.controlnet, self.features.t2iadapter, self.features.inpaint)]  


    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addCommonParams(params, diffusers_params)
        if(not self.features.img2img):
            self.addTxt2ImgParams(params, diffusers_params)
        if(self.features.img2img):
            self.addImg2ImgParams(params, diffusers_params)
        if(self.features.controlnet or self.features.t2iadapter):
            self.addConditioningImageParams(params, diffusers_params)
        if(self.features.ipadapter):
            self.addIpAdapterParams(params, diffusers_params)
        if(self.features.inpaint):
            self.addInpaintParams(params, diffusers_params)
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.images[0], seed
