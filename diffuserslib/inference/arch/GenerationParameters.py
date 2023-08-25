from typing import List
import PIL.Image

IMAGETYPE_INITIMAGE = "initimage"
IMAGETYPE_MASKIMAGE = "maskimage"
IMAGETYPE_CONTROLIMAGE = "controlimage"



class ModelParameters:
    def __init__(self, name:str, weight:float = 1.0):
        self.name = name
        self.weight = weight


class ControlImageParameters:
    def __init__(self, 
                 image:PIL.Image.Image, 
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


    def getMaskImage(self) -> ControlImageParameters|None:
        for controlimage in self.controlimages:
            if(controlimage.type == IMAGETYPE_MASKIMAGE):
                return controlimage
        return None
    
    def getInitImage(self) -> ControlImageParameters|None:
        for controlimage in self.controlimages:
            if(controlimage.type == IMAGETYPE_INITIMAGE):
                return controlimage
        return None
    
    def getControlImages(self) -> List[ControlImageParameters]:
        controlimages = []
        for controlimage in self.controlimages:
            if(controlimage.type == IMAGETYPE_CONTROLIMAGE):
                controlimages.append(controlimage)
        return controlimages

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
        


class TileGenerationParameters(GenerationParameters):
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0
