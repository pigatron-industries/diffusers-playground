from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines, MAX_SEED
from diffuserslib.inference.GenerationParameters import GenerationParameters, ControlImageType, ControlImageParameters
from diffuserslib.inference.DiffusersUtils import tiledImageProcessor
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .TileSizeCalculatorNode import TileSizeCalculatorNode
from PIL import Image
import random
import copy
from imgcat import imgcat


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

        print(f"ImageDiffusionTiledNode: tilesize = {tilesize}, overlap = {tileoverlap}")

        masktile = conditioning_inputs_tile[0] if len(conditioning_inputs_tile) > 0 else None
        print("ImageDiffusionTiledNode: Using mask tile:")
        if(masktile is not None):
            imgcat(masktile.image)

        outimage, seed = self.tiledGeneration(params=params, masktile=masktile, tilewidth=tilesize[0], tileheight=tilesize[1], overlap=tileoverlap)

        if(isinstance(outimage, Image.Image)):
            return outimage
        else:
            raise Exception("Output is not an image")
        

    @staticmethod
    def tiledGeneration(params:GenerationParameters, tilewidth=768, tileheight=768, overlap=128, masktile:ControlImageParameters|None=None, callback=None):
        initimageparams = params.getInitImage()
        if initimageparams is not None:
            initimage = initimageparams.image
        else:
            initimage = None

        controlimages = [ controlimageparams.image for controlimageparams in params.getImages(ControlImageType.IMAGETYPE_CONTROLIMAGE) ]
        if(params.seed is None):
            params.seed = random.randint(0, MAX_SEED)
        
        def imageToImageFunc(initimagetile:Image.Image|None, controlimagetiles:List[Image.Image]):
            assert(DiffusersPipelines.pipelines is not None)
            tileparams = copy.deepcopy(params)
            if(initimagetile is not None):
                tileparams.width = initimagetile.width
                tileparams.height = initimagetile.height
            elif(controlimagetiles is not None and len(controlimagetiles) > 0):
                tileparams.width = controlimagetiles[0].width
                tileparams.height = controlimagetiles[0].height
            tileparams.generationtype = "generate"
            if(initimagetile is not None):
                tileparams.setInitImage(initimagetile)
            print("ImageDiffusionTiledNode: Tile init image:")
            # imgcat(initimagetile)
            if masktile is not None:
                tileparams.controlimages.append(masktile)
            for i in range(len(controlimagetiles)):
                tileparams.setControlImage(i, controlimagetiles[i])
            image, _ = DiffusersPipelines.pipelines.generate(tileparams)
            print("ImageDiffusionTiledNode: Tile output image:")
            # imgcat(image)
            return image
        
        return tiledImageProcessor(processor=imageToImageFunc, initimage=initimage, controlimages=controlimages, tilewidth=tilewidth, tileheight=tileheight, overlap=overlap, callback=callback), params.seed




    def callback(self, status:str, totalslices:int, slicesdone:int, finished_slice:Image.Image|None = None):
        self.total_slices = totalslices
        self.slices_done = slicesdone
        self.finished_slice = finished_slice


    def getProgress(self) -> WorkflowProgress|None:
        if(self.total_slices > 0):
            return WorkflowProgress(self.slices_done/self.total_slices, self.finished_slice)
        else:
            return WorkflowProgress(0, None)