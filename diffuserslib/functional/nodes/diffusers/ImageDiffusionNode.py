from ...FunctionalNode import FunctionalNode
from ...FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .RandomPromptProcessorNode import RandomPromptProcessorNode
from ....inference.DiffusersPipelines import DiffusersPipelines
from ....inference.GenerationParameters import GenerationParameters, ModelParameters
from PIL import Image

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]


class ImageDiffusionNode(FunctionalNode):

    SCHEDULERS = [
        "DPMSolverMultistepScheduler", "EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"
    ]

    def __init__(self,
                 models:ModelsFuncType = [],
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
                prompt:str, 
                negprompt:str, 
                steps:int, 
                cfgscale:float, 
                seed:int|None, 
                scheduler:str,
                conditioning_inputs:List[ConditioningInputType]|None = None) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = GenerationParameters(
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            prompt=prompt,
            negprompt=prompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )

        # output, seed = DiffusersPipelines.pipelines.generate(params)
        output = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        if(isinstance(output, Image.Image)):
            return output
        else:
            raise Exception("Output is not an image")
        