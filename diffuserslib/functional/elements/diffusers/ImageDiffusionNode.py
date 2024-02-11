from ...FunctionalNode import FunctionalNode, TypeInfo
from ...FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncType
from ....inference.DiffusersPipelines import DiffusersPipelines
from ....inference.GenerationParameters import GenerationParameters, ModelParameters

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]


ConditioningInputFuncsType = List[ConditioningInputType] | List[Callable[[], ConditioningInputType]]

class ImageDiffusionNode(FunctionalNode):
    def __init__(self,
                 pipelines:DiffusersPipelines,
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
        self.addParam("models", models, TypeInfo("Model.generate", multiple=True))
        self.addParam("prompt", prompt, TypeInfo("String"))
        self.addParam("negprompt", negprompt, TypeInfo("String"))
        self.addParam("steps", steps, TypeInfo("Int"))
        self.addParam("cfgscale", cfgscale, TypeInfo("Float"))
        self.addParam("size", size, TypeInfo("Size"))
        self.addParam("seed", seed, TypeInfo("Int"))
        self.addParam("scheduler", scheduler, TypeInfo("String"))
        self.addParam("conditioning_inputs", conditioning_inputs, TypeInfo("ConditioningInput", multiple=True))
        self.pipelines = pipelines


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
        params = GenerationParameters(
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            prompt=prompt,
            negprompt=negprompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )
        output, seed = self.pipelines.generate(params)
        if(type(output) == Image.Image):
            return output
        else:
            raise Exception("Output is not an image")
        