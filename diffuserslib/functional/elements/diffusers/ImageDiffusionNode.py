from ...FunctionalNode import FunctionalNode, TypeInfo
from ...FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .RandomPromptProcessorNode import RandomPromptProcessorNode
from ....inference.DiffusersPipelines import DiffusersPipelines
from ....inference.GenerationParameters import GenerationParameters, ModelParameters

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]


class ImageDiffusionNode(FunctionalNode):
    name = "Image Diffusion"

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
        self.addParam("models", models, TypeInfo("Model.generate", multiple=True))
        self.addParam("prompt", prompt, TypeInfo(ParamType.FREETEXT))
        self.addParam("negprompt", negprompt, TypeInfo(ParamType.FREETEXT))
        self.addParam("steps", steps, TypeInfo(ParamType.INT))
        self.addParam("cfgscale", cfgscale, TypeInfo(ParamType.FLOAT))
        self.addParam("size", size, TypeInfo(ParamType.IMAGE_SIZE))
        self.addParam("seed", seed, TypeInfo(ParamType.INT))
        self.addParam("scheduler", scheduler, TypeInfo(ParamType.STRING, restrict_choice=self.SCHEDULERS))
        self.addParam("conditioning_inputs", conditioning_inputs, TypeInfo("ConditioningInput", multiple=True))
        self.prompt_processor = RandomPromptProcessorNode()


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
            prompt=self.prompt_processor.process(prompt, False),
            negprompt=self.prompt_processor.process(negprompt, False),
            steps=steps,
            cfgscale=cfgscale,
            seed=seed,
            scheduler=scheduler,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )

        output, seed = DiffusersPipelines.pipelines.generate(params)
        if(type(output) == Image.Image):
            return output
        else:
            raise Exception("Output is not an image")
        