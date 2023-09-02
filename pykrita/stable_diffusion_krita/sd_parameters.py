from typing import List
from dataclasses import dataclass, fields, field, asdict
import json

IMAGETYPE_INITIMAGE = "initimage"
IMAGETYPE_MASKIMAGE = "maskimage"
IMAGETYPE_CONTROLIMAGE = "controlimage"


GENERATIONTYPE_TXT2IMG = "txt2img"
GENERATIONTYPE_IMG2IMG = "img2img"
GENERATIONTYPE_INPAINT = "inpaint"
GENERATIONTYPE_UPSCALE = "upscale"
GENERATIONTYPE_CONTROLNET_SUFFIX = "_controlnet"


@dataclass
class ModelParameters:
    name:str
    weight:float = 1.0


@dataclass
class ControlImageParameters:
    image64:str = ""
    type:str = IMAGETYPE_INITIMAGE
    model:"str|None" = None
    condscale:float = 1.0


@dataclass
class GenerationParameters:
    generationtype:"str|None" = None
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
    seed:"int|None" = None
    scheduler:str = "DPMSolverMultistepScheduler"
    models:List[ModelParameters] = field(default_factory=list)
    tiling:bool = False
    controlimages:List[ControlImageParameters] = field(default_factory=list)

    # Tiling parameters
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0

    # Upscale parameters
    upscalemethod:str = "esrgan"
    upscaleamount:int = 4

    @classmethod
    def fromJson(cls, jsonbytes:bytes):
        dict = json.loads(jsonbytes)
        return cls.fromDict(dict)
    
    @classmethod
    def fromDict(cls, dict:dict):
        if("models" in dict):
            dict["models"] = [ModelParameters(**model) for model in dict["models"]]
        if("controlimages" in dict):
            dict["controlimages"] = [ControlImageParameters(**controlimage) for controlimage in dict["controlimages"]]
        return cls(**dict)

    def getImage(self, type:str) -> "ControlImageParameters|None":
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

    def getMaskImage(self) -> "ControlImageParameters|None":
        return self.getImage(IMAGETYPE_MASKIMAGE)
    
    def getInitImage(self) -> "ControlImageParameters|None":
        return self.getImage(IMAGETYPE_INITIMAGE)
    
    def getControlImages(self) -> List[ControlImageParameters]:
        return self.getImages(IMAGETYPE_CONTROLIMAGE)

    def getGenerationType(self) -> str:
        if(self.generationtype is not None):
            return self.generationtype
        else:
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
        
    def setImage(self, image64:str, type:str):
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimage.image64 = image64
                return
        self.controlimages.append(ControlImageParameters(image64=image64, type=type))

    def setInitImage(self, image64:str):
        self.setImage(image64, IMAGETYPE_INITIMAGE)

    def setMaskImage(self, image64:str):
        self.setImage(image64, IMAGETYPE_MASKIMAGE)
                
    def setControlImage(self, index:int, image64:str):
        for controlimage in self.controlimages:
            if(controlimage.type == IMAGETYPE_CONTROLIMAGE):
                if(index == 0):
                    controlimage.image64 = image64
                    return
                index -= 1
        raise Exception("Control image index out of range")
    
    def toJson(self):
        return json.dumps(self, cls=DataclassEncoder, indent=2)
    
    def copyValuesFrom(self, obj):
        for field in fields(self):
            if(hasattr(obj, field.name)):
                setattr(self, field.name, getattr(obj, field.name))


class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (ModelParameters, ControlImageParameters, GenerationParameters)):
            return asdict(obj)
        return super().default(obj)