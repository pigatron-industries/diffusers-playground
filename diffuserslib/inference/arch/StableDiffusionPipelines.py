from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageParameters, IMAGETYPE_MASKIMAGE, IMAGETYPE_INITIMAGE, IMAGETYPE_CONTROLIMAGE
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
    

class StableDiffusionTextToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        # self.custom_pipeline = 'composable_stable_diffusion'
        # self.custom_pipeline = 'lpw_stable_diffusion'
        self.custom_pipeline = StableDiffusionPipeline
        super().__init__(self.custom_pipeline, params, device)

    def inference(self, params:GenerationParameters):
        # args = {}
        # if (self.custom_pipeline == 'composable_stable_diffusion'):
        #     prompts = prompt.split("|")
        #     weights = None
        #     if (len(prompts) > 1):
        #         negprompt = [negprompt] * len(prompts)
        #         weights = []
        #         for promptpart in prompts:
        #             weight = promptpart.split(" ")[-1]
        #             if (weight.isnumeric()):
        #                 weights.append(weight)
        #             else:
        #                 weights.append("1")
        #         weights = " | ".join(weights)
        #     args['weights'] = weights

        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, width=params.width, height=params.height, seed=params.seed, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionImageToImagePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        super().__init__(StableDiffusionImg2ImgPipeline, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 strength=params.strength, scheduler=params.scheduler)


class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        super().__init__(DiffusionPipeline, params, device)

    def inference(self, params:GenerationParameters):
        initimage = params.controlimages[0].image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler)


class StableDiffusionControlNetPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device, cls=DiffusionPipeline):
        controlmodelids = self.getConditioningModels(params)
        conditioningtype = self.getConditioningType(params)
        if(conditioningtype == "t2iadapter"):
            conditioningmodels = self.createConditioningModels(controlmodelids, T2IAdapter)
            super().__init__(params=params, device=device, cls=cls, adapter=conditioningmodels)
        elif(conditioningtype == "controlnet"):
            conditioningmodels = self.createConditioningModels(controlmodelids, ControlNetModel)
            super().__init__(params=params, device=device, cls=cls, controlnet=conditioningmodels)
        

class StableDiffusionTextToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        self.conditioningtype = self.getConditioningType(params)
        if(self.conditioningtype == "t2iadapter"):
            super().__init__(cls=StableDiffusionAdapterPipeline, params=params, device=device)
        elif(self.conditioningtype == "controlnet"):
            super().__init__(cls=StableDiffusionControlNetPipeline, params=params, device=device)

    def inference(self, params:GenerationParameters):
        condscales = self.getConditioningScales(params)
        conditioningimages = self.getConditioningImages(params)
        if(self.conditioningtype == "t2iadapter"):
            return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=conditioningimages, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler, adapter_conditioning_scale=condscales)
        elif(self.conditioningtype == "controlnet"):
            return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=conditioningimages, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler, controlnet_conditioning_scale=condscales)

    

class StableDiffusionImageToImageControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, device, params:GenerationParameters):
        super().__init__(cls=StableDiffusionControlNetImg2ImgPipeline, params=params, device=device)

    def inference(self, params:GenerationParameters):
        condscales = self.getConditioningScales(params)
        conditioningimages = self.getConditioningImages(params)
        initimage = params.getInitImage().image.convert("RGB")
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, control_image=conditioningimages, 
                                 guidance_scale=params.cfgscale, strength=params.strength, scheduler=params.scheduler, controlnet_conditioning_scale=condscales)
    

# Standard stable diffusion inpaint pipeline
# class StableDiffusionInpaintPipelineWrapper(StableDiffusionPipelineWrapper):
#     def __init__(self, preset:DiffusersModel, device):
#         super().__init__(StableDiffusionInpaintPipeline, preset, device)

#     def inference(self, prompt, negprompt, seed, initimage, maskimage, scale, steps, scheduler, strength=1.0, **kwargs):
#         initimage = initimage.convert("RGB")
#         maskimage = maskimage.convert("RGB")
#         return super().inference(prompt=prompt, negative_prompt=negprompt, seed=seed, image=initimage, mask_image=maskimage, guidance_scale=scale, num_inference_steps=steps, 
#                                  strength=strength, scheduler=scheduler, width=initimage.width, height=initimage.height)


class StableDiffusionInpaintPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        # The inpaint controlnet model is supposed to make non inpaint models work with inpaint
        params.controlimages.insert(0, ControlImageParameters(type=IMAGETYPE_CONTROLIMAGE, model=INPAINT_CONTROL_MODEL))
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, params=params, device=device)

    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None):
            raise ValueError("Must provide both initimage and maskimage")
        initimage = initimageparams.image.convert("RGB")
        maskimage = maskimageparams.image.convert("RGB")
        inpaint_pt = make_inpaint_condition(initimage=initimage, maskimage=maskimage)
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, control_image=inpaint_pt, 
                                 guidance_scale=params.cfgscale, num_inference_steps=params.steps, strength=params.strength, scheduler=params.scheduler, width=initimage.width, height=initimage.height)
    

class StableDiffusionInpaintControlNetPipelineWrapper(StableDiffusionControlNetPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        # The inpaint controlnet model is supposed to make non inpaint models work with inpaint
        params.controlimages.insert(0, ControlImageParameters(type=IMAGETYPE_CONTROLIMAGE, model=INPAINT_CONTROL_MODEL))
        super().__init__(cls=StableDiffusionControlNetInpaintPipeline, params=params, device=device)

    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None):
            raise ValueError("Must provide both initimage and maskimage")
        initimage = initimageparams.image.convert("RGB")
        maskimage = maskimageparams.image.convert("RGB")
        condscales = self.getConditioningScales(params)
        conditioningimages = self.getConditioningImages(params)
        return super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, mask_image=maskimage, 
                                 control_image=conditioningimages, guidance_scale=params.cfgscale, num_inference_steps=params.steps, scheduler=params.scheduler, 
                                 width=initimage.width, height=initimage.height, controlnet_conditioning_scale=condscales)
    

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