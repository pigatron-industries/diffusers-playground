from ..GenerationParameters import GenerationParameters
from ...models.DiffusersModelPresets import DiffusersModel
from ...StringUtils import mergeDicts
from typing import Tuple
from PIL import Image
import sys
import random
import torch


MAX_SEED = 4294967295


class DiffusersPipelineWrapper:
    def __init__(self, params:GenerationParameters, inferencedevice:str):
        self.params = params
        self.inferencedevice = inferencedevice

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
    

    # Useful function to get conditioning models and images

    def getConditioningType(self, params:GenerationParameters):
        conditioningtype = None
        for controlimage in params.getControlImages():
            if(controlimage.model is not None):
                # TODO relying on the word t2iadapter in model name is not ideal
                if("adapter" in controlimage.model):
                    currentconditioningtype = "t2iadapter"
                else:
                    currentconditioningtype = "controlnet"
                if conditioningtype is not None and currentconditioningtype != conditioningtype:
                    raise ValueError("Cannot mix t2iadapter and controlnet conditioning")
                conditioningtype = currentconditioningtype
        return conditioningtype

    def createConditioningModels(self, conditioningmodelids:List[str], conditioningClass=ControlNetModel):
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