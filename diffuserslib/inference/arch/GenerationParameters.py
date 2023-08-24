from typing import List
import PIL.Image


class ModelParameters:
    name:str = ""
    weight:float = 1.0


class ControlImageParameters:
    image:PIL.Image.Image|None = None
    model:str = ""
    condscale:float = 1.0


class GenerationParameters:
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
    models:List[ModelParameters] = []
    tiling:bool = False


class InpainGenerationParameters(GenerationParameters):
    maskimage:PIL.Image.Image|None = None


class TileGenerationParameters(GenerationParameters):
    tilemethod:str = "singlepass"
    tilealignmentx:str = "tile_centre"
    tilealignmenty:str = "tile_centre"
    tilewidth:int = 512
    tileheight:int = 512
    tileoverlap:int = 0
