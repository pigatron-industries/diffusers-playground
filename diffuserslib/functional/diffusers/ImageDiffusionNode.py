from ..FunctionalNode import FunctionalNode
from ..FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncType
from ...inference.DiffusersPipelines import DiffusersPipelines
from ...inference.GenerationParameters import GenerationParameters, ModelParameters

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
        self.pipelines = pipelines
        args = {
            "models": models,
            "prompt": prompt,
            "negprompt": negprompt,
            "steps": steps,
            "cfgscale": cfgscale,
            "size": size,
            "seed": seed,
            "scheduler": scheduler,
            "conditioning_inputs": conditioning_inputs
        }
        super().__init__("image_diffusion", args)


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
        