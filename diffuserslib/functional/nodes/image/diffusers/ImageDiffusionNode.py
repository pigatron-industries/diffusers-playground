from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters, ModelParameters, LoraParameters
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from PIL import Image

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]
LorasType = List[LoraParameters]
LorasFuncType = LorasType | Callable[[], LorasType]

class ImageDiffusionNode(FunctionalNode):

    SCHEDULERS = [
        "DDIMScheduler", "DPMSolverMultistepScheduler", "DPMSolverSDEScheduler", "EDMDPMSolverMultistepScheduler", "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"
    ]

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType|None = None,
                 prompt:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 sigmas:FloatsFuncType|None = None,
                 clipskip:IntFuncType|None = None,
                 conditioning_inputs:ConditioningInputFuncsType|None = None,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("prompt", prompt, str)
        self.addParam("negprompt", negprompt, str)
        self.addParam("steps", steps, int)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed", seed, int)
        self.addParam("scheduler", scheduler, str)
        self.addParam("sigmas", sigmas, List[float])
        self.addParam("clipskip", clipskip, int)
        self.addParam("conditioning_inputs", conditioning_inputs, List[ConditioningInputType])


    def process(self, 
                size:SizeType, 
                models:ModelsType, 
                loras:LorasType,
                prompt:str, 
                negprompt:str, 
                steps:int, 
                cfgscale:float, 
                seed:int|None, 
                scheduler:str,
                sigmas:List[float]|None,
                clipskip:int|None,
                conditioning_inputs:List[ConditioningInputType]|None) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        conditioningparams = []
        if(conditioning_inputs is not None):
            for conditioning_input in conditioning_inputs:
                if(conditioning_input.image is not None):
                    conditioningparams.append(conditioning_input)
                    width = conditioning_input.image.width
                    height = conditioning_input.image.height

        if(size is not None):
            width = size[0]
            height = size[1]
        
        params = GenerationParameters(
            safetychecker=False,
            width=width,
            height=height,
            models=models,
            loras=loras,
            prompt=prompt,
            negprompt=negprompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            sigmas=sigmas,
            clipskip=clipskip,
            controlimages=conditioningparams
        )

        print(params)

        output, seed = DiffusersPipelines.pipelines.generate(params)
        # output = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        if(isinstance(output, Image.Image)):
            return output
        else:
            raise Exception("Output is not an image")
        