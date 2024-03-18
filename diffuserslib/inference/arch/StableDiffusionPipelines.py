from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from diffuserslib.inference.LORA import LORA
from PIL import Image
from diffusers import ( # Pipelines
                        DiffusionPipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline,
                        StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline,
                        StableDiffusionAdapterPipeline, AnimateDiffPipeline, PIAPipeline, AnimateDiffVideoToVideoPipeline,
                        # Conditioning models
                        T2IAdapter, ControlNetModel, MotionAdapter,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, UniPCMultistepScheduler, LCMScheduler)
import torch
import sys
import numpy as np
from compel import Compel
from typing import List
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType



def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class StableDiffusionPipelineWrapper(DiffusersPipelineWrapper):
    INPAINT_CONTROL_MODEL = "lllyasviel/control_v11p_sd15_inpaint"
    LCM_LORA_MODEL = "latent-consistency/lcm-lora-sdv1-5"

    def __init__(self, cls, params:GenerationParameters, device):
        print(f"creating pipeline {cls.__name__ if type(cls) is type else cls}")
        if(params.modelConfig is None):
            raise ValueError("Must provide modelConfig")
        self.safety_checker = params.safetychecker
        self.device = device
        self.lora_names = []
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params, cls)
        super().__init__(params, inferencedevice)


    def createPipeline(self, params:GenerationParameters, cls):
        if(params.modelConfig is None):
            raise ValueError("Must provide modelConfig")
        pipeline_params = self.createPipelineParams(params)
        if(type(cls) is str):
            pipeline_params['custom_pipeline'] = cls
            cls = DiffusionPipeline
        if (params.modelConfig.modelpath.endswith('.safetensors') or params.modelConfig.modelpath.endswith('.ckpt')):
            self.pipeline = cls.from_single_file(params.modelConfig.modelpath, load_safety_checker=self.safety_checker, **pipeline_params).to(self.device)
        else:
            # CLIP skip implementation, but breaks lora loading
            # text_encoder = CLIPTextModel.from_pretrained(preset.modelpath, subfolder="text_encoder", num_hidden_layers=11)
            # self.pipeline = cls.from_pretrained(preset.modelpath, text_encoder=text_encoder, **args).to(self.device)
            self.pipeline = cls.from_pretrained(params.modelConfig.modelpath, **pipeline_params).to(self.device)
            
        self.pipeline.enable_attention_slicing()
        # pipeline.enable_model_cpu_offload()
        # self.pipeline.enable_xformers_memory_efficient_attention()  # doesn't work on mps


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    

    def initIpAdapter(self, params:GenerationParameters):
        ipadapterparams = params.getConditioningParamsByModelType(ControlImageType.IMAGETYPE_IPADAPTER)
        repos = []
        subfolders = []
        filenames = []
        for ipadapterparam in ipadapterparams:
            if(ipadapterparam.model is not None):
                ipadaptermodelname = self.splitModelName(ipadapterparam.model)
                repos.append(ipadaptermodelname.repository)
                subfolders.append(ipadaptermodelname.subfolder)
                filenames.append(ipadaptermodelname.filename)
        self.pipeline.load_ip_adapter(repos, subfolders, filenames)


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

        if(self.initparams.modelConfig.autocast):
            with torch.autocast(self.inferencedevice):
                output = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs)
        else:
            output = self.pipeline(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, generator=generator, **kwargs)
        return output, seed
    

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


    def add_lora(self, lora:LORA):
        if(lora.name not in self.lora_names):
            self.lora_names.append(lora.name)
            self.pipeline.load_lora_weights(lora.path, adapter_name=lora.name.split('.', 1)[0])


    def add_loras(self, loras:List[LORA], weights:List[float]):
        for lora in loras:
            self.add_lora(lora)
        lora_weights = []
        lora_names = []
        for i, lora in enumerate(loras):
            lora_weights.append(weights[i])
            lora_names.append(lora.name.split('.', 1)[0])
        self.pipeline.set_adapters(lora_names, lora_weights)



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
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)

        super().__init__(params=params, device=device, cls=cls)

        if(self.features.ipadapter):
            self.initIpAdapter(params)


    def getPipelineClass(self, params:GenerationParameters):
        if(self.features.ipadapter_faceid):
            return "ip_adapter_face_id" # Experimental face id adapter community pipeline
        else:
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


class StableDiffusionUpscalePipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        super().__init__(DiffusionPipeline, params, device)

    def inference(self, params:GenerationParameters):
        initimageparams = params.getInitImage()
        if(initimageparams is None or initimageparams.image is None):
            raise ValueError("Must provide initimage")
        initimage = initimageparams.image.convert("RGB")
        output, seed = super().diffusers_inference(prompt=params.prompt, negative_prompt=params.negprompt, seed=params.seed, image=initimage, guidance_scale=params.cfgscale, 
                                 num_inference_steps=params.steps, scheduler=params.scheduler)
        return output.images[0], seed
    

class StableDiffusionAnimateDiffPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)
        self.adapter = MotionAdapter.from_pretrained("vladmandic/animatediff-v3", torch_dtype=torch.float16)
        super().__init__(cls, params, device)


    def getPipelineClass(self, params:GenerationParameters):
        if(self.features.controlnet):
            return "pipeline_animatediff_controlnet"
        elif(self.features.img2img):
            initimageparams = params.getInitImage()
            if(initimageparams is not None and isinstance(initimageparams.image, list)):
                return AnimateDiffVideoToVideoPipeline
            else:
                return "pipeline_animatediff_img2video"
        else:
            return AnimateDiffPipeline


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
            self.addInferenceParamsImg2Img(params, diffusers_params)
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


class StableDiffusionPersonalizedImageAnimatorPipelineWrapper(StableDiffusionPipelineWrapper):
    def __init__(self, params:GenerationParameters, device):
        self.adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter", torch_dtype=torch.float16)
        super().__init__(PIAPipeline, params, device)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        pipeline_params['motion_adapter'] = self.adapter
        pipeline_params['torch_dtype'] = torch.float16
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params


    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        imageparams = params.getInitImage()
        if(imageparams is not None and imageparams.image is not None):
            diffusers_params['image'] = imageparams.image
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['scheduler'] = params.scheduler
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_frames'] = params.frames
        output, seed = super().diffusers_inference(**diffusers_params)
        return output.frames[0], seed