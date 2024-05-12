from .StableDiffusionPipelines import StableDiffusionPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType
from typing import List
from PIL import Image
from diffusers import ( # Pipelines
                        StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                        StableDiffusionXLControlNetPipeline, StableDiffusionXLAdapterPipeline, StableDiffusionXLControlNetImg2ImgPipeline,
                        StableDiffusionXLControlNetInpaintPipeline,
                        # Conditioning models
                        T2IAdapter, ControlNetModel, MotionAdapter, AnimateDiffSDXLPipeline, 
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler, LCMScheduler, DPMSolverSDEScheduler)
from compel import Compel, ReturnedEmbeddingsType
import torch


class NoWatermark:
    def apply_watermark(self, img):
        return img


class StableDiffusionXLPipelineWrapper(StableDiffusionPipelineWrapper):
    LCM_LORA_MODEL = "latent-consistency/lcm-lora-sdxl"

    def __init__(self, cls, params:GenerationParameters, device, dtype=None):
        super().__init__(cls=cls, params=params, device=device, dtype=dtype)
        self.pipeline.watermark = NoWatermark()
        if(params.scheduler == "LCMScheduler"):
            self.pipeline.load_lora_weights(self.LCM_LORA_MODEL)
            self.pipeline.fuse_lora()
                

    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        if(self.initparams.scheduler == "LCMScheduler"):
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
        self.dtype = None
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)

        super().__init__(params=params, device=device, cls=cls)

        if(self.features.ipadapter):
            self.initIpAdapter(params)


    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        if(self.features.differential):
            return "pipeline_stable_diffusion_xl_differential_img2img"
        return self.PIPELINE_MAP[(self.features.img2img, self.features.controlnet, self.features.t2iadapter, self.features.inpaint)]  


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        if(self.features.controlnet):
            self.addPipelineParamsControlNet(params, pipeline_params)
        if(self.features.t2iadapter):
            self.addPipelineParamsT2IAdapter(params, pipeline_params)
        if(self.features.ipadapter):
            self.addPipelineParamsIpAdapter(params, pipeline_params)
        return pipeline_params


    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addInferenceParamsCommon(params, diffusers_params)
        if(not self.features.img2img):
            self.addInferenceParamsTxt2Img(params, diffusers_params)
        if(self.features.img2img):
            self.addInferenceParamsImg2Img(params, diffusers_params)
        if(self.features.differential):
            self.addInferenceParamsDifferential(params, diffusers_params)
        if(self.features.controlnet):
            self.addInferenceParamsControlNet(params, diffusers_params)
        if(self.features.t2iadapter):
            self.addInferenceParamsT2IAdapter(params, diffusers_params)
        if(self.features.ipadapter):
            self.addInferenceParamsIpAdapter(params, diffusers_params)
        if(self.features.inpaint):
            self.addInferenceParamsInpaint(params, diffusers_params)
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.images[0], seed



class StableDiffusionXLAnimateDiffPipelineWrapper(StableDiffusionXLPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        self.dtype = torch.float16
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)
        self.adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=self.dtype)
        super().__init__(cls, params, device, dtype = self.dtype)


    def getPipelineClass(self, params:GenerationParameters):
        return AnimateDiffSDXLPipeline


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        pipeline_params['motion_adapter'] = self.adapter
        pipeline_params['torch_dtype'] = torch.float16
        self.addPipelineParamsCommon(params, pipeline_params)
        if(self.features.controlnet):
            self.addPipelineParamsControlNet(params, pipeline_params)
        return pipeline_params


    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['scheduler'] = params.scheduler
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_frames'] = params.frames
        if(self.features.controlnet):
            self.addInferenceParamsControlNet(params, diffusers_params)
        if(self.features.img2img):
            initimageparams = params.getInitImage()
            if(initimageparams is not None):
                diffusers_params['strength'] = initimageparams.condscale
                if(isinstance(initimageparams.image, list)):
                    diffusers_params['video'] = initimageparams.image
                else:
                    diffusers_params['image'] = initimageparams.image
            diffusers_params['latent_interpolation_method'] = "slerp"  # "slerp" or "lerp"
        if(self.features.ipadapter):
            self.addInferenceParamsIpAdapter(params, diffusers_params)
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.frames[0], seed
    

    def addInferenceParamsControlNet(self, params:GenerationParameters, diffusers_params):
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        videos = []
        scales = []
        for controlnetparam in controlnetparams:
            if(controlnetparam.image is not None):
                video = controlnetparam.image
                videos.append(video)
                scales.append(controlnetparam.condscale)
        diffusers_params['conditioning_frames'] = videos
        diffusers_params['controlnet_conditioning_scale'] = scales