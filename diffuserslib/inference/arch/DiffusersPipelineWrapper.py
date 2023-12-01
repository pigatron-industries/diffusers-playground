from ..GenerationParameters import GenerationParameters, ControlImageType
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Tuple, List
from PIL import Image
from dataclasses import dataclass
import sys
import random
import torch


MAX_SEED = 4294967295

@dataclass
class PipelineFeatures:
    img2img = False
    controlnet = False
    t2iadapter = False
    inpaint = False
    ipadapter = False


@dataclass
class SplitModelName:
    repository:str
    subfolder:str
    filename:str


class DiffusersPipelineWrapper:
    def __init__(self, params:GenerationParameters, inferencedevice:str):
        self.params = params
        self.inferencedevice = inferencedevice
        self.features = self.getPipelineFeatures(params)

    def inference(self, params:GenerationParameters) -> Tuple[Image.Image, int]: # type: ignore
        pass

    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed
    
    def paramsMatch(self, params:GenerationParameters) -> bool:
        match = (self.params.getGenerationType() == self.params.getGenerationType() and 
                 len(self.params.models) == len(params.models) and
                 len(self.params.loras) == len(params.loras) and
                 len(self.params.controlimages) == len(params.controlimages))
        if(not match):
           return False

        for i in range(len(self.params.models)):
            if(self.params.models[i].name != params.models[i].name or self.params.models[i].weight != params.models[i].weight):
                return False

        for i in range(len(self.params.loras)):
            if(self.params.loras[i].name != params.loras[i].name or self.params.loras[i].weight != params.loras[i].weight):
                return False
            
        for i in range(len(self.params.controlimages)):
            if(self.params.controlimages[i].type != params.controlimages[i].type and self.params.controlimages[i].model != params.controlimages[i].model):
                return False
            
        return True

    def add_embeddings(self, token, embeddings):
        raise ValueError(f"add_embeddings not implemented for pipeline")
    

    # Functions for adding inference parameters

    def addCommonParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['scheduler'] = params.scheduler

    def addTxt2ImgParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_inference_steps'] = params.steps

    def addImg2ImgParams(self, params:GenerationParameters, diffusers_params):
        initimage = params.getInitImage()
        if(initimage is not None and initimage.image is not None):
            diffusers_params['image'] = initimage.image.convert("RGB")
            diffusers_params['strength'] = params.strength

    def addInpaintParams(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None or initimageparams.image is None or maskimageparams.image is None):
            raise ValueError("Must provide both initimage and maskimage")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['mask_image'] = maskimageparams.image.convert("RGB")
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['strength'] = params.strength
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height

    def addIpAdapterParams(self, params:GenerationParameters, diffusers_params):
        ipadapterparams = params.getIpAdapterImage()
        if(ipadapterparams is None or ipadapterparams.image is None):
            raise ValueError("Must provide ipadapter image")
        diffusers_params['ip_adapter_image'] = ipadapterparams.image.convert("RGB")

    def addConditioningImageParams(self, params:GenerationParameters, diffusers_params):
        condscales = self.getConditioningScales(params)
        conditioningimages = self.getConditioningImages(params)
        diffusers_params['width'] = conditioningimages[0].width
        diffusers_params['height'] = conditioningimages[0].height
        if(self.features.img2img):
            diffusers_params['control_image'] = conditioningimages
        else:
            diffusers_params['image'] = conditioningimages
        if(self.features.controlnet):
            diffusers_params['controlnet_conditioning_scale'] = condscales
        else:
            diffusers_params['adapter_conditioning_scale'] = condscales


    # Useful functions to get conditioning models and images

    def getPipelineFeatures(self, params:GenerationParameters):
        features = PipelineFeatures()
        for conditioningimage in params.controlimages:
            if(conditioningimage.type == ControlImageType.IMAGETYPE_INITIMAGE):
                features.img2img = True
            if(conditioningimage.type == ControlImageType.IMAGETYPE_MASKIMAGE):
                features.inpaint = True
            if(conditioningimage.type == ControlImageType.IMAGETYPE_IPADAPTER):
                features.ipadapter = True
            # TODO relying on the word t2iadapter in model name is not ideal
            if(conditioningimage.modelConfig is not None and 'control' in conditioningimage.modelConfig.modelid):
                features.controlnet = True
            elif(conditioningimage.modelConfig is not None and 'adapter' in conditioningimage.modelConfig.modelid):
                features.t2iadapter = True
        return features

    def createConditioningModels(self, conditioningmodelids:List[str], conditioningClass):
        conditioningmodels = []
        for conditioningmodelid in conditioningmodelids:
            conditioningmodels.append(conditioningClass.from_pretrained(conditioningmodelid))
        if(len(conditioningmodels) == 1):
            conditioningmodels = conditioningmodels[0]
        return conditioningmodels
    
    def getConditioningModels(self, params:GenerationParameters):
        conditioningmodelids = []
        for controlimageparams in params.getControlImages():
            conditioningmodelids.append(controlimageparams.model)
        return conditioningmodelids
    
    def getConditioningScales(self, params:GenerationParameters):
        condscales = []
        for controlimage in params.getControlImages():
            condscales.append(controlimage.condscale)
        if len(condscales) == 1:
            return condscales[0]
        return condscales
    
    def getConditioningImages(self, params:GenerationParameters):
        conditioningimages = []
        for conditioningimage in params.getControlImages():
            colourspace = "RGB"
            if ("colourspace" in conditioningimage.modelConfig.data):
                colourspace = conditioningimage.modelConfig.data["colourspace"]
            conditioningimages.append(conditioningimage.image.convert(colourspace))
        return conditioningimages
    
    def splitModelName(self, modelname:str) -> SplitModelName:
        splitname = modelname.split('/')
        if(len(splitname) == 2):
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '', '')
        elif(len(splitname) == 3):
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '', splitname[2])
        else:
            return SplitModelName(f'{splitname[0]}/{splitname[1]}', '/'.join(splitname[2:-1]), splitname[-1])
