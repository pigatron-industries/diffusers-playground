from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters, TiledGenerationParameters
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from PIL import Image


class ImageDiffusionDetailerNode(FunctionalNode):

    def __init__(self,
                 image:ImageFuncType,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 prompt:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("prompt", prompt, str)
        self.addParam("negprompt", negprompt, str)
        self.addParam("steps", steps, int)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed", seed, int)
        self.addParam("scheduler", scheduler, str)


    def process(self, 
                image:Image.Image,
                models:ModelsType, 
                loras:LorasType,
                prompt:str, 
                negprompt:str, 
                steps:int, 
                cfgscale:float, 
                seed:int|None, 
                scheduler:str) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = GenerationParameters(
            safetychecker=False,
            models=models,
            loras=loras,
            prompt=prompt,
            negprompt=negprompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler
        )

        print(params)

        output, seed = DiffusersPipelines.pipelines.generate(params)
        # output = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        if(isinstance(output, Image.Image)):
            return output
        else:
            raise Exception("Output is not an image")
        