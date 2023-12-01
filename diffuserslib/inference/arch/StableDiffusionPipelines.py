from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, IMAGETYPE_MASKIMAGE, IMAGETYPE_INITIMAGE, IMAGETYPE_CONTROLIMAGE
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Callable, List
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline,
                        StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline,
                        StableDiffusionAdapterPipeline, 
                        # Conditioning models
                        T2IAdapter, ControlNetModel,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler, LCMScheduler)
from transformers import CLIPTextModel
import torch
import sys
import numpy as np
from compel import Compel


INPAINT_CONTROL_MODEL = "lllyasviel/control_v11p_sd15_inpaint"


def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    LCM_LORA_MODEL = "latent-consistency/lcm-lora-sdv1-5"

    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params.modelConfig, cls, **kwargs)
        super().__init__(params, inferencedevice)

    def createPipeline(self, preset:DiffusersModel, cls, **kwargs):
        args = self.createPipelineArgs(preset, **kwargs)
        if (preset.modelpath.endswith('.safetensors') or preset.modelpath.endswith('.ckpt')):
            self.pipeline = cls.from_single_file(preset.modelpath, load_safety_checker=self.safety_checker, **args).to(self.device)
        else:
            # CLIP skip implementation, but breaks lora loading
            # text_encoder = CLIPTextModel.from_pretrained(preset.modelpath, subfolder="text_encoder", num_hidden_layers=11)
            # self.pipeline = cls.from_pretrained(preset.modelpath, text_encoder=text_encoder, **args).to(self.device)
            self.pipeline = cls.from_pretrained(preset.modelpath, **args).to(self.device)
            
        self.pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()
        # pipeline.enable_xformers_memory_efficient_attention()

    def createPipelineArgs(self, preset:DiffusersModel, **kwargs):
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
        return schedulerClass
    
    def diffusers_inference(self, prompt, negative_prompt, seed, scheduler=None, tiling=False, **kwargs):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler)
        self.pipeline.vae.enable_tiling(tiling)

        compel = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder)
        conditioning = compel(prompt)
        negative_conditioning = compel(negative_prompt)
        # [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

        if(self.params.modelConfig.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs).images[0]
        else:
            image = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs).images[0]
        return image, seed
    
    def add_embeddings(self, token, embeddings):
        self.add_embedding_to_text_encoder(token, embeddings[0], self.pipeline.tokenizer, self.pipeline.text_encoder)

    def add_embedding_to_text_encoder(self, token, embedding, tokenizer, text_encoder):
        dtype = self.pipeline.text_encoder.get_input_embeddings().weight.dtype
        for i, embedding_vector in enumerate(embedding):
            #  add token for each vector in embedding
            tokenpart = token + str(i)
            embedding_vector.to(dtype)
            num_added_tokens = tokenizer.add_tokens(tokenpart)
            if(num_added_tokens == 0):
                raise ValueError(f"The tokenizer already contains the token {tokenpart}")
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_id = tokenizer.convert_tokens_to_ids(tokenpart)
            text_encoder.get_input_embeddings().weight.data[token_id] = embedding_vector
    


class StableDiffusionGeneratePipelineWrapper(StableDiffusionPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    controlnet, t2iadapter, inpaint
        (False,     False,      False,      False):    StableDiffusionPipeline,
        (False,     True,       False,      False):    StableDiffusionControlNetPipeline,
        (False,     False,      True,       False):    StableDiffusionAdapterPipeline,
        (True,      False,      False,      False):    StableDiffusionImg2ImgPipeline,
        (True,      True,       False,      False):    StableDiffusionControlNetImg2ImgPipeline,
        (True,      False,      False,      True):     StableDiffusionInpaintPipeline,
        (True,      True,       False,      True):     StableDiffusionControlNetInpaintPipeline,
    }

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        if(not self.is_controlnet and not self.is_t2iadapter):
            super().__init__(params=params, device=device, cls=cls)
        else:
            controlmodelids = self.getConditioningModels(params)
            conditioningtype = self.getConditioningType(params)
            if(conditioningtype == "t2iadapter"):
                conditioningmodels = self.createConditioningModels(controlmodelids, T2IAdapter)
                super().__init__(params=params, device=device, cls=cls, adapter=conditioningmodels)
            elif(conditioningtype == "controlnet"):
                conditioningmodels = self.createConditioningModels(controlmodelids, ControlNetModel)
                super().__init__(params=params, device=device, cls=cls, controlnet=conditioningmodels)

    def getPipelineClass(self, params:GenerationParameters):
        self.is_img2img = False
        self.is_controlnet = False
        self.is_t2iadapter = False
        self.is_inpaint = False
        for conditioningimage in params.controlimages:
            if(conditioningimage.type == IMAGETYPE_INITIMAGE):
                self.is_img2img = True
            if(conditioningimage.type == IMAGETYPE_MASKIMAGE):
                self.is_inpaint = True
            elif(conditioningimage.modelConfig is not None and 'control' in conditioningimage.modelConfig.modelid):
                self.is_controlnet = True
            elif(conditioningimage.modelConfig is not None and 'adapter' in conditioningimage.modelConfig.modelid):
                self.is_t2iadapter = True
        return self.PIPELINE_MAP[(self.is_img2img, self.is_controlnet, self.is_t2iadapter, self.is_inpaint)]


    def addCommonParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['scheduler'] = params.scheduler
    
    def addImg2ImgParams(self, params:GenerationParameters, diffusers_params):
        initimage = params.getInitImage()
        if(initimage is not None and initimage.image is not None):
            diffusers_params['image'] = initimage.image.convert("RGB")
            diffusers_params['strength'] = params.strength

    def addTxt2ImgParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_inference_steps'] = params.steps

    def addConditioningImageParams(self, params:GenerationParameters, diffusers_params):
        condscales = self.getConditioningScales(params)
        conditioningimages = self.getConditioningImages(params)
        diffusers_params['width'] = conditioningimages[0].width
        diffusers_params['height'] = conditioningimages[0].height
        if(self.is_img2img):
            diffusers_params['control_image'] = conditioningimages
        else:
            diffusers_params['image'] = conditioningimages
        if(self.is_controlnet):
            diffusers_params['controlnet_conditioning_scale'] = condscales
        else:
            diffusers_params['adapter_conditioning_scale'] = condscales

    def addInpaintParams(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None or initimageparams.image is None or maskimageparams.image is None):
            raise ValueError("Must provide both initimage and maskimage")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['mask_image'] = maskimageparams.image.convert("RGB")
        # TODO need to append inpaint conditioning to other conditioning
        # diffusers_params['control_image'] = make_inpaint_condition(initimage=initimage, maskimage=maskimage)
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['strength'] = params.strength
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height
    

    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addCommonParams(params, diffusers_params)
        if(not self.is_img2img):
            self.addTxt2ImgParams(params, diffusers_params)
        if(self.is_img2img):
            self.addImg2ImgParams(params, diffusers_params)
        if(self.is_controlnet or self.is_t2iadapter):
            self.addConditioningImageParams(params, diffusers_params)
        if(self.is_inpaint):
            self.addInpaintParams(params, diffusers_params)
        return super().diffusers_inference(**diffusers_params)


class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        super().__init__(DiffusionPipeline, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler)


def make_inpaint_condition(initimage:Image.Image, maskimage:Image.Image) -> torch.Tensor:
    npinitimage = np.array(initimage.convert("RGB")).astype(np.float32) / 255.0
    npmaskimage = np.array(maskimage.convert("L")).astype(np.float32) / 255.0
    npinitimage[npmaskimage > 0.5] = -1.0  # set as masked pixel
    npinitimage = np.expand_dims(npinitimage, 0).transpose(0, 3, 1, 2)
    npinitimage = torch.from_numpy(npinitimage)
    return npinitimage


def pil_to_pt(image:Image.Image) -> torch.Tensor:
    npimage = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    npimage = np.expand_dims(npimage, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(npimage)