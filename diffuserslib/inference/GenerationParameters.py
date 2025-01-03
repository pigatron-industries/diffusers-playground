from typing import List
from PIL import Image
from dataclasses import field
from pydantic.dataclasses import dataclass
from ..ImageUtils import base64DecodeImage
from ..models.DiffusersModelPresets import DiffusersModel, DiffusersModelType
import json, inspect
from pydantic import BaseModel, ConfigDict
import yaml


class ControlImageType:
    IMAGETYPE_INITIMAGE = "initimage"
    IMAGETYPE_MASKIMAGE = "maskimage"
    IMAGETYPE_DIFFMASKIMAGE = "diffmaskimage"
    IMAGETYPE_IPADAPTER = "ipadapter"
    IMAGETYPE_IPADAPTER_FACEID = "ipadapter_faceid"
    IMAGETYPE_CONTROLIMAGE = "controlimage"


GENERATIONTYPE_TXT2IMG = "txt2img"
GENERATIONTYPE_IMG2IMG = "img2img"
GENERATIONTYPE_INPAINT = "inpaint"
GENERATIONTYPE_UPSCALE = "upscale"


class ModelConfig:
    arbitrary_types_allowed = True


@dataclass
class ModelParameters:
    name:str
    weight:float = 1.0
    base:str|None = None

def modelparameters_representer(dumper: yaml.Dumper, data: ModelParameters) -> yaml.Node:
    return dumper.represent_dict(data.__dict__)
yaml.add_representer(ModelParameters, modelparameters_representer)


@dataclass
class LoraParameters:
    name:str
    weight:float = 1.0

def loraparameters_representer(dumper: yaml.Dumper, data: LoraParameters) -> yaml.Node:
    return dumper.represent_dict(data.__dict__)
yaml.add_representer(LoraParameters, loraparameters_representer)


@dataclass(config=ModelConfig)
class ControlImageParameters:
    image:Image.Image|List[Image.Image]|None = None
    image64:str = ""
    type:str = ControlImageType.IMAGETYPE_INITIMAGE
    preprocessor:str|None = None
    model:str|None = None
    condscale:float = 1.0
    # runtime 
    modelConfig:DiffusersModel|None = None

    def __post_init__(self):
        if(self.image64 is not None and self.image64 != ""):
            self.image = base64DecodeImage(self.image64)


def controlimageparameters_representer(dumper: yaml.Dumper, data: ControlImageParameters) -> yaml.Node:
    return dumper.represent_dict(data.__dict__)
yaml.add_representer(ControlImageParameters, controlimageparameters_representer)


@dataclass(config=ModelConfig)
class GenerationParameters:
    generationtype:str = "generate"
    batch:int = 1
    prescale:float = 1.0
    safetychecker:bool = True
    prompt:str = ""
    negprompt:str = ""
    steps:int = 40
    cfgscale:float = 7.0
    strength:float = 1.0
    width:int = 512
    height:int = 512
    frames:int = 16
    seed:int|None = None
    scheduler:str = "DPMSolverMultistepScheduler"
    sigmas:List[float]|None = None
    models:List[ModelParameters] = field(default_factory=list)
    loras:List[LoraParameters] = field(default_factory=list)
    tiling:bool = False
    clipskip:int|None = None
    controlimages:List[ControlImageParameters] = field(default_factory=list)
    # runtime
    modelConfig:List[DiffusersModel] = field(default_factory=list)

    @classmethod
    def from_json(cls, jsonbytes:bytes):
        dict = json.loads(jsonbytes)
        if("models" in dict):
            dict["models"] = [ModelParameters(**model) for model in dict["models"]]
        if("controlimages" in dict):
            dict["controlimages"] = [ControlImageParameters(**controlimage) for controlimage in dict["controlimages"]]
        if("loras" in dict):
            dict["loras"] = [LoraParameters(**lora) for lora in dict["loras"]]
        return cls.from_dict(dict)

    @classmethod
    def from_dict(cls, dict:dict):
        """ Create a GenerationParameters object from a dictionary, ignoring any keys that are not parameters of the class """
        return cls(**{
            k: v for k, v in dict.items() 
            if k in inspect.signature(cls).parameters
        })

    def __post_init__(self):
        self.original_prompt:str = self.prompt
        self.original_negprompt:str = self.negprompt

    def getImage(self, type:str) -> ControlImageParameters|None:
        for controlimage in self.controlimages:
            if(controlimage.type == type or (controlimage.modelConfig is not None and type in controlimage.modelConfig.modeltypes)):
                return controlimage
        return None
    
    def getImages(self, type:str) -> List[ControlImageParameters]:
        controlimages = []
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimages.append(controlimage)
        return controlimages

    def getMaskImage(self) -> ControlImageParameters|None:
        return self.getImage(ControlImageType.IMAGETYPE_MASKIMAGE)

    def getInitImage(self) -> ControlImageParameters|None:
        return self.getImage(ControlImageType.IMAGETYPE_INITIMAGE)
    
    def getConditioningParamsByModelType(self, modeltype:str) -> List[ControlImageParameters]:
        conditioningparams = []
        for conditioningparam in self.controlimages:
            if(conditioningparam.modelConfig is not None and modeltype in conditioningparam.modelConfig.modeltypes):
                conditioningparams.append(conditioningparam)
        return conditioningparams
    
    def setImage(self, image:Image.Image, type:str):
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimage.image = image
                return
        self.controlimages.append(ControlImageParameters(image=image, type=type))

    def setInitImage(self, image:Image.Image):
        self.setImage(image, ControlImageType.IMAGETYPE_INITIMAGE)

    def setMaskImage(self, image:Image.Image):
        self.setImage(image, ControlImageType.IMAGETYPE_MASKIMAGE)
                
    def setControlImage(self, index:int, image:Image.Image):
        for controlimage in self.controlimages:
            if(controlimage.type == ControlImageType.IMAGETYPE_CONTROLIMAGE):
                if(index == 0):
                    controlimage.image = image
                    return
                index -= 1
        raise Exception("Control image index out of range")


def generationparameters_representer(dumper: yaml.Dumper, data: GenerationParameters) -> yaml.Node:
    dict = data.__dict__.copy()
    del dict['modelConfig']
    return dumper.represent_dict(dict)
yaml.add_representer(GenerationParameters, generationparameters_representer)


@dataclass
class VideoGenerationParameters(GenerationParameters):
    fps:int = 7


@dataclass(config=ModelConfig)
class TiledGenerationParameters(GenerationParameters):
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0


@dataclass(config=ModelConfig)
class UpscaleGenerationParameters(GenerationParameters):
    upscalemethod:str = "esrgan"
    upscaleamount:int = 4
