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

class ImageDiffusionNode(FunctionalNode):

    SCHEDULERS = [
        "DDIMScheduler", "DPMSolverMultistepScheduler", "EDMDPMSolverMultistepScheduler", "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"
    ]

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType = (512, 512),
                 prompt:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
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
        self.addParam("conditioning_inputs", conditioning_inputs, ConditioningInputType)


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
                conditioning_inputs:List[ConditioningInputType]|None) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        conditioningparams = []
        if(conditioning_inputs is not None):
            for conditioning_input in conditioning_inputs:
                if(conditioning_input.image is not None):
                    conditioningparams.append(conditioning_input)
        
        params = GenerationParameters(
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            loras=loras,
            prompt=prompt,
            negprompt=negprompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            controlimages=conditioningparams
        )

        output, seed = DiffusersPipelines.pipelines.generate(params)
        # output = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        if(isinstance(output, Image.Image)):
            return output
        else:
            raise Exception("Output is not an image")
        