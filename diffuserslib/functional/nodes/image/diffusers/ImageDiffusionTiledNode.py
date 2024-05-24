from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters
from diffuserslib.inference.DiffusersUtils import tiledProcessorCentred, tiledGeneration
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .TileSizeCalculatorNode import TileSizeCalculatorNode
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
                 conditioning_inputs_tile:ConditioningInputFuncsType = [],
                 tilesize:SizeFuncType|None = None,
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
        self.addParam("conditioning_inputs_tile", conditioning_inputs_tile, List[ConditioningInputType])
        self.addParam("tilesize", tilesize, Tuple[int, int])
        self.addParam("tileoverlap", tileoverlap, int)
        self.total_slices = 0
        self.slices_done = 0
        self.finished_slice = None


    def process(self, 
                models:ModelsType, 
                conditioning_inputs:List[ConditioningInputType],
                conditioning_inputs_tile:List[ConditioningInputType],
                tilesize:Tuple[int, int]|None,
                tileoverlap:int,
                **kwargs) -> Image.Image:
        self.total_slices = 0
        self.slices_done = 0
        self.finished_slice = None
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        if(len(conditioning_inputs) == 0):
            raise Exception("No conditioning inputs provided")
        
        conditioningparams = []
        if(conditioning_inputs is not None):
            for conditioning_input in conditioning_inputs:
                if(conditioning_input.image is not None and isinstance(conditioning_input.image, Image.Image) and conditioning_input.condscale > 0):
                    conditioningparams.append(conditioning_input)
                    width = conditioning_input.image.width
                    height = conditioning_input.image.height

        params = GenerationParameters(safetychecker=False, models=models, controlimages=conditioningparams, **kwargs)
        if(tilesize is None):    
            tilewidth = TileSizeCalculatorNode.calcTileSize(width, 1152, tileoverlap)
            tileheight = TileSizeCalculatorNode.calcTileSize(height, 1152, tileoverlap)
            tilesize = (tilewidth, tileheight)

        masktile = conditioning_inputs_tile[0] #TODO allow multiple mask tiles
        outimage, seed = tiledGeneration(pipelines=DiffusersPipelines.pipelines, params=params, masktile=masktile, tilewidth=tilesize[0], tileheight=tilesize[1], overlap=tileoverlap)

        if(isinstance(outimage, Image.Image)):
            return outimage
        else:
            raise Exception("Output is not an image")
        

    def callback(self, status:str, totalslices:int, slicesdone:int, finished_slice:Image.Image|None = None):
        self.total_slices = totalslices
        self.slices_done = slicesdone
        self.finished_slice = finished_slice


    def getProgress(self) -> WorkflowProgress|None:
        if(self.total_slices > 0):
            return WorkflowProgress(self.slices_done/self.total_slices, self.finished_slice)
        else:
            return WorkflowProgress(0, None)