from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types import *
from diffuserslib.functional.nodes.image.diffusers.ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import ModelParameters, LoraParameters
from diffuserslib.inference.arch.StableVideoDiffusionPipelines import VideoGenerationParameters
from PIL import Image

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]
LorasType = List[LoraParameters]
LorasFuncType = LorasType | Callable[[], LorasType]

class VideoDiffusionNode(FunctionalNode):

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType = (512, 512),
                 seed:IntFuncType|None = None,
                 frames:IntFuncType = 16,
                 fps:IntFuncType = 7,
                 prompt:StringFuncType|None = None,
                 conditioning_inputs:ConditioningInputFuncsType|None = None,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("seed", seed, int)
        self.addParam("frames", frames, int)
        self.addParam("fps", fps, int)
        self.addParam("prompt", prompt, str)
        self.addParam("conditioning_inputs", conditioning_inputs, ConditioningInputType)


    def process(self, 
                size:SizeType, 
                models:ModelsType, 
                loras:LorasType, 
                seed:int|None,
                frames:int,
                fps:int,
                prompt:str|None = None,
                conditioning_inputs:List[ConditioningInputType]|None = None) -> List[Image.Image]:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = VideoGenerationParameters(
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            loras=loras,
            seed=seed,
            frames=frames,
            fps=fps,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )

        output, seed = DiffusersPipelines.pipelines.generate(params)
        if(isinstance(output, List)):
            return output
        else:
            raise Exception("Output is not a list of images")
        