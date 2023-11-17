from typing import List
from PIL import Image
from dataclasses import dataclass, field
from ..ImageUtils import base64DecodeImage
from ..models.DiffusersModelPresets import DiffusersModel, DiffusersConditioningModel
import json, inspect

IMAGETYPE_INITIMAGE = "initimage"
IMAGETYPE_MASKIMAGE = "maskimage"
IMAGETYPE_CONTROLIMAGE = "controlimage"


GENERATIONTYPE_TXT2IMG = "txt2img"
GENERATIONTYPE_IMG2IMG = "img2img"
GENERATIONTYPE_INPAINT = "inpaint"
GENERATIONTYPE_UPSCALE = "upscale"
GENERATIONTYPE_CONTROLNET_SUFFIX = "_controlnet"


@dataclass(unsafe_hash=True)
class ModelParameters:
    name:str
    weight:float = 1.0


@dataclass(unsafe_hash=True)
class LoraParameters:
    name:str
    weight:float = 1.0


@dataclass(unsafe_hash=True)
class ControlImageParameters:
    image:Image.Image|None = None
    image64:str = ""
    type:str = IMAGETYPE_INITIMAGE
    preprocessor:str|None = None
    model:str|None = None
    condscale:float = 1.0
    # extra 
    modelConfig:DiffusersModel|None = None

    def __post_init__(self):
        if(self.image64 is not None and self.image64 != ""):
            self.image = base64DecodeImage(self.image64)


@dataclass(unsafe_hash=True)
class GenerationParameters:
    generationtype:str|None = None
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
    seed:int|None = None
    scheduler:str = "DPMSolverMultistepScheduler"
    models:List[ModelParameters] = field(default_factory=list)
    loras:List[LoraParameters] = field(default_factory=list)
    tiling:bool = False
    controlimages:List[ControlImageParameters] = field(default_factory=list)
    # extras
    modelConfig:DiffusersModel|None = None

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

    def getImage(self, type:str) -> ControlImageParameters|None:
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                return controlimage
        return None
    
    def getImages(self, type:str) -> List[ControlImageParameters]:
        controlimages = []
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimages.append(controlimage)
        return controlimages

    def getMaskImage(self) -> ControlImageParameters|None:
        return self.getImage(IMAGETYPE_MASKIMAGE)
    
    def getInitImage(self) -> ControlImageParameters|None:
        return self.getImage(IMAGETYPE_INITIMAGE)
    
    def getControlImages(self) -> List[ControlImageParameters]:
        return self.getImages(IMAGETYPE_CONTROLIMAGE)

    def getGenerationType(self) -> str:
        # if(self.generationtype is not None):
        #     return self.generationtype
        # else:
        generationtype = ""
        if(self.getMaskImage() is not None):
            generationtype += GENERATIONTYPE_INPAINT
        elif(self.getInitImage() is not None):
            generationtype += GENERATIONTYPE_IMG2IMG
        else:
            generationtype += GENERATIONTYPE_TXT2IMG
        if(len(self.getControlImages()) > 0):
            generationtype += GENERATIONTYPE_CONTROLNET_SUFFIX
        return generationtype
        
    def setImage(self, image:Image.Image, type:str):
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimage.image = image
                return
        self.controlimages.append(ControlImageParameters(image=image, type=type))

    def setInitImage(self, image:Image.Image):
        self.setImage(image, IMAGETYPE_INITIMAGE)

    def setMaskImage(self, image:Image.Image):
        self.setImage(image, IMAGETYPE_MASKIMAGE)
                
    def setControlImage(self, index:int, image:Image.Image):
        for controlimage in self.controlimages:
            if(controlimage.type == IMAGETYPE_CONTROLIMAGE):
                if(index == 0):
                    controlimage.image = image
                    return
                index -= 1
        raise Exception("Control image index out of range")


@dataclass
class TiledGenerationParameters(GenerationParameters):
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0


@dataclass
class UpscaleGenerationParameters(GenerationParameters):
    upscalemethod:str = "esrgan"
    upscaleamount:int = 4
