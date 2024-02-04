from ..FunctionalNode import FunctionalNode
from ..FunctionalTyping import *
from ...inference.DiffusersPipelines import DiffusersPipelines
from ...inference.GenerationParameters import GenerationParameters, ModelParameters

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]


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
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler"):
        self.pipelines = pipelines
        args = {
            "models": models,
            "prompt": prompt,
            "negprompt": negprompt,
            "steps": steps,
            "cfgscale": cfgscale,
            "size": size,
            "seed": seed,
            "scheduler": scheduler
        }
        super().__init__(args)


    def process(self, 
                size:SizeType, 
                models:ModelsType, 
                prompt:str, 
                negprompt:str, 
                steps:int, 
                cfgscale:float, 
                seed:int|None, 
                scheduler:str) -> Image.Image:
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
            scheduler=scheduler
        )
        output = self.pipelines.generate(params)
        if(type(output) == Image.Image):
            return output
        else:
            raise Exception("Output is not an image")
        