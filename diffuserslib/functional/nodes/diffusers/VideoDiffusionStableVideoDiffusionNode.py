from ...FunctionalNode import FunctionalNode
from ...types.FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from ....inference.DiffusersPipelines import DiffusersPipelines
from ....inference.GenerationParameters import GenerationParameters, ModelParameters, LoraParameters
from PIL import Image

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]
LorasType = List[LoraParameters]
LorasFuncType = LorasType | Callable[[], LorasType]

class VideoDiffusionStableVideoDiffusionNode(FunctionalNode):

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType = (512, 512),
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 frames:IntFuncType = 16,
                 conditioning_inputs:ConditioningInputFuncsType|None = None,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("steps", steps, int)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed", seed, int)
        self.addParam("scheduler", scheduler, str)
        self.addParam("frames", frames, int)
        self.addParam("conditioning_inputs", conditioning_inputs, ConditioningInputType)


    def process(self, 
                size:SizeType, 
                models:ModelsType, 
                loras:LorasType,
                steps:int, 
                cfgscale:float, 
                seed:int|None, 
                scheduler:str,
                frames:int,
                conditioning_inputs:List[ConditioningInputType]|None = None) -> List[Image.Image]:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = GenerationParameters(
            generationtype="animatediff",
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            loras=loras,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            frames=frames,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )

        output, seed = DiffusersPipelines.pipelines.generate(params)
        if(isinstance(output, List)):
            return output
        else:
            raise Exception("Output is not a list of images")
        