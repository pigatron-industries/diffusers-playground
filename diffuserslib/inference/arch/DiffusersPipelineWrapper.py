from ..GenerationParameters import GenerationParameters, ControlImageType
from ...models.DiffusersModelPresets import DiffusersModel, DiffusersModelType
from ...StringUtils import mergeDicts
from ...ImageUtils import pilToCv2
from typing import Tuple, List, Dict
from PIL import Image
from dataclasses import dataclass
from diffusers import DiffusionPipeline, T2IAdapter, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from torchvision import transforms
import sys
import random
import torch
import numpy as np

import cv2
from insightface.app import FaceAnalysis


MAX_SEED = 4294967295

@dataclass
class PipelineFeatures:
    img2img = False
    controlnet = False
    t2iadapter = False
    inpaint = False
    ipadapter = False
    ipadapter_faceid = False
    differential = False


@dataclass
class SplitModelName:
    repository:str
    subfolder:str
    filename:str


class DiffusersPipelineWrapper:
    def __init__(self, params:GenerationParameters, inferencedevice:str, cls, controlnet_cls:type = ControlNetModel, dtype = None, **kwargs):
        if(dtype is None):
            dtype = torch.bfloat16
        self.lora_names = []
        self.controlnet_cls = controlnet_cls
        self.dtype = dtype
        self.initparams = params
        self.inferencedevice = inferencedevice
        self.features = self.getPipelineFeatures(params)
        self.createPipeline(params, cls, **kwargs)


    def createPipeline(self, params:GenerationParameters, cls):
        pipeline_params = self.createPipelineParams(params)
        if(type(cls) is str):
            pipeline_params['custom_pipeline'] = cls
            cls = DiffusionPipeline

        for i, modelConfig in enumerate(params.modelConfig):
            pipeline = self.loadPipeline(modelConfig, cls, pipeline_params)
            if(i == 0):
                self.pipeline = pipeline
            else:
                self.mergeModel(pipeline, params.models[i].weight)
            
        if("torch_dtype" in pipeline_params and pipeline_params["torch_dtype"] not in [torch.float16, torch.bfloat16]):
            self.pipeline.enable_attention_slicing()


    def loadPipeline(self, modelConfig, cls, pipelineParams):
        if (modelConfig.modelpath.endswith('.safetensors') or modelConfig.modelpath.endswith('.ckpt')):
            return cls.from_single_file(modelConfig.modelpath, **pipelineParams).to(self.device)
        else:
            return cls.from_pretrained(modelConfig.modelpath, **pipelineParams).to(self.device)


    def mergeModel(self, mergePipeline, weight):
        for moduleName in self.pipeline.config.keys():
            module1 = getattr(self.pipeline, moduleName)
            module2 = getattr(mergePipeline, moduleName)
            if hasattr(module1, "state_dict") and hasattr(module2, "state_dict"):
                print(f"Merging state_dict of {moduleName}")
                updateStateDictFunc = getattr(module1, "load_state_dict")
                theta_0 = getattr(module1, "state_dict")()
                theta_1 = getattr(module2, "state_dict")()
                for key in theta_0.keys():
                    if key in theta_1:
                        theta_0[key] = (1 - weight) * theta_0[key] + weight * theta_1[key]
                for key in theta_1.keys():
                    if key not in theta_0:
                        theta_0[key] = theta_1[key]
                updateStateDictFunc(theta_0)


    def createPipelineParams(self, params:GenerationParameters) -> Dict:
        raise NotImplementedError("createPipelineParams not implemented for pipeline")


    def inference(self, params:GenerationParameters) -> Tuple[Image.Image, int]:
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
        output, seed = self.diffusers_inference(**diffusers_params)
        return output.images[0], seed
    

    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, scheduler=None, **kwargs):
        raise NotImplementedError("diffusers_inference not implemented for pipeline")


    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed
    
    def paramsMatch(self, params:GenerationParameters) -> bool:
        match = (self.initparams.generationtype == self.initparams.generationtype and 
                 len(self.initparams.models) == len(params.models) and
                 len(self.initparams.loras) == len(params.loras) and
                 len(self.initparams.controlimages) == len(params.controlimages))
        if(not match):
           return False

        for i in range(len(self.initparams.models)):
            if(self.initparams.models[i].name != params.models[i].name or self.initparams.models[i].weight != params.models[i].weight):
                return False

        for i in range(len(self.initparams.loras)):
            if(self.initparams.loras[i].name != params.loras[i].name or self.initparams.loras[i].weight != params.loras[i].weight):
                return False
            
        for i in range(len(self.initparams.controlimages)):
            if(self.initparams.controlimages[i].type != params.controlimages[i].type or self.initparams.controlimages[i].model != params.controlimages[i].model):
                return False
            
        return True


    def interrupt(self):
        self.pipeline._interrupt = True


    def add_embeddings(self, token, embeddings):
        raise ValueError(f"add_embeddings not implemented for pipeline")
    

    def add_lora(self, lora):
        if(lora.name not in self.lora_names):
            self.lora_names.append(lora.name)
            self.pipeline.load_lora_weights(lora.path, adapter_name=lora.name.split('.', 1)[0])


    def add_loras(self, loras, weights:List[float]):
        for lora in loras:
            self.add_lora(lora)
        lora_weights = []
        lora_names = []
        for i, lora in enumerate(loras):
            lora_weights.append(weights[i])
            lora_names.append(lora.name.split('.', 1)[0])

        if(hasattr(self.pipeline, 'set_adapters')):
            if(len(lora_names) > 0):
                self.pipeline.set_adapters(lora_names, lora_weights)


    ### Functions for adding pipeline and inference parameters

    def addPipelineParamsCommon(self, params:GenerationParameters, pipeline_params):
        modelConfig = params.modelConfig[0]
        if (not params.safetychecker):
            pipeline_params['safety_checker'] = None
        if(modelConfig.revision is not None):
            pipeline_params['revision'] = modelConfig.revision
            if(modelConfig.revision == 'fp16'):
                pipeline_params['torch_dtype'] = torch.float16
        if(self.dtype is not None):
            pipeline_params['torch_dtype'] = self.dtype
        if(modelConfig.vae is not None):
            pipeline_params['vae'] = AutoencoderKL.from_pretrained(modelConfig.vae, 
                                                        torch_dtype=pipeline_params['torch_dtype'] if 'torch_dtype' in pipeline_params else None, 
                                                        revision=pipeline_params['revision'] if 'revision' in pipeline_params else None)
        return pipeline_params
    
    def addPipelineParamsControlNet(self, params:GenerationParameters, pipeline_params):
        args = {}
        if(self.dtype is not None):
            args['torch_dtype'] = self.dtype
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        controlnet = []
        for controlnetparam in controlnetparams:
            controlnet.append(self.controlnet_cls.from_pretrained(controlnetparam.model, **args))
        pipeline_params['controlnet'] = controlnet
        return pipeline_params
    
    def addPipelineParamsT2IAdapter(self, params:GenerationParameters, pipeline_params):
        args = {}
        if(self.dtype is not None):
            args['torch_dtype'] = self.dtype
        t2iadapterparams = params.getConditioningParamsByModelType(DiffusersModelType.t2iadapter)
        t2iadapter = []
        for t2iadapterparam in t2iadapterparams:
            t2iadapter.append(T2IAdapter.from_pretrained(t2iadapterparam.model), **args)
        pipeline_params['adapter'] = t2iadapter
        return pipeline_params
    
    def addPipelineParamsIpAdapter(self, params:GenerationParameters, pipeline_params):
        args = {}
        if(self.dtype is not None):
            args['torch_dtype'] = self.dtype
        pipeline_params['image_encoder'] = CLIPVisionModelWithProjection.from_pretrained(
                                                "h94/IP-Adapter", 
                                                subfolder="models/image_encoder",
                                                **args
                                            )
        return pipeline_params
    

    def addInferenceParamsCommon(self, params:GenerationParameters, diffusers_params):
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['scheduler'] = params.scheduler
        diffusers_params['clip_skip'] = params.clipskip
        if(params.sigmas is not None):
            diffusers_params['sigmas'] = params.sigmas

    def addInferenceParamsTxt2Img(self, params:GenerationParameters, diffusers_params):
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_inference_steps'] = params.steps

    def addInferenceParamsImg2Img(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        if(initimageparams is not None and initimageparams.image is not None):
            diffusers_params['image'] = initimageparams.image.convert("RGB")
            diffusers_params['strength'] = initimageparams.condscale

    def addInferenceParamsDifferential(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getImage(ControlImageType.IMAGETYPE_DIFFMASKIMAGE)
        if(initimageparams is not None and maskimageparams is not None):
            inittensor = transforms.ToTensor()(initimageparams.image.convert("RGB"))
            inittensor = inittensor * 2 - 1
            inittensor = inittensor.unsqueeze(0).to(self.inferencedevice)
            maskimage = maskimageparams.image.convert("L")
            maskimage = Image.eval(maskimage, lambda x: 255 - x)
            masktensor = transforms.ToTensor()(maskimage)
            masktensor = masktensor.to(self.inferencedevice)
            diffusers_params['image'] = inittensor.to(self.device)
            diffusers_params['original_image'] = inittensor.to(self.device)
            diffusers_params['map'] = masktensor.to(self.device)

    def addInferenceParamsInpaint(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None or initimageparams.image is None or maskimageparams.image is None):
            raise ValueError("Must provide both initimage and maskimage")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['mask_image'] = maskimageparams.image.convert("RGB")
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['strength'] = initimageparams.condscale
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height


    def addInferenceParamsIpAdapter(self, params:GenerationParameters, diffusers_params):
        ipadapterparams = params.getConditioningParamsByModelType(DiffusersModelType.ipadapter)
        images = []
        scales = []
        for ipadapterparam in ipadapterparams:
            images.append(ipadapterparam.image.convert("RGB"))
            scales.append(ipadapterparam.condscale)
        diffusers_params['ip_adapter_image'] = images
        self.pipeline.set_ip_adapter_scale(scales)
        # if(ipadapterparams.modelConfig.preprocess == "faceid"):
        #     diffusers_params['image_embeds'] = self.preprocessFaceEmbeds(ipadapterparams.image.convert("RGB"))
        # else:


    def preprocessFaceEmbeds(self, face_image):
        faceanalysis = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        faceanalysis.prepare(ctx_id=0, det_size=(640, 640))
        image = np.asarray(face_image)
        faces = faceanalysis.get(image)
        image_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        return image_embeds


    def addInferenceParamsControlNet(self, params:GenerationParameters, diffusers_params):
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        images = []
        scales = []
        for controlnetparam in controlnetparams:
            if(controlnetparam.image is not None and controlnetparam.condscale > 0):
                images.append(controlnetparam.image.convert("RGB"))
                scales.append(controlnetparam.condscale)
        diffusers_params['control_image' if self.features.img2img else 'image'] = images
        diffusers_params['controlnet_conditioning_scale'] = scales


    def addInferenceParamsT2IAdapter(self, params:GenerationParameters, diffusers_params):
        t2iadapterparams = params.getConditioningParamsByModelType(DiffusersModelType.t2iadapter)
        images = []
        scales = []
        for t2iadapterparam in t2iadapterparams:
            if(t2iadapterparam.image is not None):
                images.append(t2iadapterparam.image.convert("RGB"))
                scales.append(t2iadapterparam.condscale)
        diffusers_params['image'] = images
        diffusers_params['adapter_conditioning_scale'] = scales


    # Useful functions to get conditioning models and images

    def getPipelineFeatures(self, params:GenerationParameters):
        features = PipelineFeatures()
        for conditioningimage in params.controlimages:
            if(conditioningimage.type == ControlImageType.IMAGETYPE_INITIMAGE):
                features.img2img = True
            elif(conditioningimage.type == ControlImageType.IMAGETYPE_MASKIMAGE):
                features.inpaint = True
            elif(conditioningimage.type == ControlImageType.IMAGETYPE_DIFFMASKIMAGE):
                features.differential = True
            elif(conditioningimage.modelConfig is not None):
                if(DiffusersModelType.ipadapter in conditioningimage.modelConfig.modeltypes):
                    features.ipadapter = True
                    if(conditioningimage.modelConfig.preprocess == "faceid"):
                        features.ipadapter_faceid = True    
                elif(DiffusersModelType.controlnet in conditioningimage.modelConfig.modeltypes):
                    features.controlnet = True
                elif(DiffusersModelType.t2iadapter in conditioningimage.modelConfig.modeltypes):
                    features.t2iadapter = True
            else:
                raise ValueError(f"Unknown control image type {conditioningimage.type}")
        return features


    def createConditioningModels(self, conditioningmodelids:List[str], conditioningClass):
        conditioningmodels = []
        for conditioningmodelid in conditioningmodelids:
            conditioningmodels.append(conditioningClass.from_pretrained(conditioningmodelid))
        if(len(conditioningmodels) == 1):
            conditioningmodels = conditioningmodels[0]
        return conditioningmodels

    
    def splitModelName(self, modelname:str) -> SplitModelName:
        splitname = modelname.split('/')
        if(len(splitname) == 2):
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '', '')
        elif(len(splitname) == 3):
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '', splitname[2])
        else:
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '/'.join(splitname[2:-1]), splitname[-1])
