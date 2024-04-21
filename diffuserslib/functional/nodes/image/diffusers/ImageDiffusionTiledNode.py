from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters
from diffuserslib.inference.DiffusersUtils import tiledProcessorCentred, tiledGeneration
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from PIL import Image


class ImageDiffusionTiledNode(FunctionalNode):

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 prompt:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 conditioning_inputs:ConditioningInputFuncsType = [],
                 tileoverlap:IntFuncType = 128,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("prompt", prompt, str)
        self.addParam("negprompt", negprompt, str)
        self.addParam("steps", steps, int)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed", seed, int)
        self.addParam("scheduler", scheduler, str)
        self.addParam("conditioning_inputs", conditioning_inputs, List[ConditioningInputType])
        self.addParam("tileoverlap", tileoverlap, int)


    def process(self, 
                models:ModelsType, 
                conditioning_inputs:List[ConditioningInputType],
                tileoverlap:int,
                **kwargs) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        if(len(conditioning_inputs) == 0):
            raise Exception("No conditioning inputs provided")
        
        conditioningparams = []
        if(conditioning_inputs is not None):
            for conditioning_input in conditioning_inputs:
                if(conditioning_input.image is not None and isinstance(conditioning_input.image, Image.Image)):
                    conditioningparams.append(conditioning_input)
                    width = conditioning_input.image.width
                    height = conditioning_input.image.height

        params = GenerationParameters(safetychecker=False, models=models, controlimages=conditioning_inputs, **kwargs)       
        tilewidth = self.calcTileSize(width, 1152, tileoverlap)
        tileheight = self.calcTileSize(height, 1152, tileoverlap)
        outimage, seed = tiledGeneration(pipelines=DiffusersPipelines.pipelines, params=params, tilewidth=tilewidth, tileheight=tileheight, overlap=tileoverlap)

        if(isinstance(outimage, Image.Image)):
            return outimage
        else:
            raise Exception("Output is not an image")
        

    def calcTileSize(self, total_length:int, max_tile_length:int, tile_overlap:int, restrict:str|None = None) -> int:
        if restrict is not None and restrict == "even":
            n = 2
        else:
            n = 1
        while True:
            tile_length = (total_length + (n - 1) * tile_overlap) // n
            if tile_length <= max_tile_length:
                tile_length = (tile_length + 7) // 8 * 8  # Round up to the nearest multiple of 8
                return tile_length
            n += 2 if restrict is not None else 1