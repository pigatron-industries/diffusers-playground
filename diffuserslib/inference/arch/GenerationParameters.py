from typing import List
from PIL import Image

IMAGETYPE_INITIMAGE = "initimage"
IMAGETYPE_MASKIMAGE = "maskimage"
IMAGETYPE_CONTROLIMAGE = "controlimage"



class ModelParameters:
    def __init__(self, name:str, weight:float = 1.0):
        self.name = name
        self.weight = weight


class ControlImageParameters:
    def __init__(self, 
                 image:Image.Image, 
                 type:str = IMAGETYPE_INITIMAGE,
                 model:str|None = None, 
                 condscale:float = 1.0):
        self.image = image
        self.type = type
        self.model = model
        self.condscale = condscale  # only affects controlnet images


class GenerationParameters:
    def __init__(self, 
                 generationtype:str|None = None,
                 safetychecker:bool = True,
                 prompt:str = "",
                 negprompt:str = "",
                 steps:int = 40,
                 cfgscale:float = 7.0,
                 strength:float = 1.0,
                 width:int = 512,
                 height:int = 512,
                 seed:int|None = None,
                 scheduler:str = "DPMSolverMultistepScheduler",
                 models:List[ModelParameters] = [],
                 tiling:bool = False,
                 controlimages:List[ControlImageParameters] = []):
        self.generationtype = generationtype
        self.safetychecker = safetychecker
        self.original_prompt = prompt
        self.prompt = prompt
        self.negprompt = negprompt
        self.steps = steps
        self.cfgscale = cfgscale
        self.strength = strength
        self.width = width
        self.height = height
        self.seed = seed
        self.scheduler = scheduler
        self.models = models
        self.tiling = tiling
        self.controlimages = controlimages

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
        if(self.generationtype is not None):
            return self.generationtype
        else:
            generationtype = ""
            if(self.getMaskImage() is not None):
                generationtype += "inpaint"
            elif(self.getInitImage() is not None):
                generationtype += "img2img"
            else:
                generationtype += "txt2img"
            if(len(self.getControlImages()) > 0):
                generationtype += "_controlnet"
            return generationtype
        
    def setImage(self, image:Image.Image, type:str):
        for controlimage in self.controlimages:
            if(controlimage.type == type):
                controlimage.image = image
                return
        self.controlimages.append(ControlImageParameters(image, type))

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



class TileGenerationParameters(GenerationParameters):
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0
