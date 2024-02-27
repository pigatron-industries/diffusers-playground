from ...FunctionalNode import FunctionalNode
from ...FunctionalTyping import *
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from ....inference.DiffusersPipelines import DiffusersPipelines
from ....inference.GenerationParameters import GenerationParameters
from PIL import Image
from .latentblending.blending_engine import BlendingEngine
import random


class LatentBlendingNode(FunctionalNode):

    MAX_SEED = 2**32 - 1

    def __init__(self,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType = (512, 512),
                 prompt1:StringFuncType = "",
                 prompt2:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 cfgscale:FloatFuncType = 7.0,
                 seed1:IntFuncType|None = None,
                 seed2:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 conditioning_inputs:ConditioningInputFuncsType|None = None,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("size", size, SizeType)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("prompt1", prompt1, str)
        self.addParam("prompt2", prompt2, str)
        self.addParam("negprompt", negprompt, str)
        self.addParam("steps", steps, int)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed1", seed1, int)
        self.addParam("seed2", seed2, int)
        self.addParam("scheduler", scheduler, str)
        self.addParam("conditioning_inputs", conditioning_inputs, ConditioningInputType)


    def process(self, 
                size:SizeType, 
                models:ModelsType, 
                loras:LorasType,
                prompt1:str, 
                prompt2:str, 
                negprompt:str, 
                steps:int, 
                cfgscale:float, 
                seed1:int|None, 
                seed2:int|None,
                scheduler:str,
                conditioning_inputs:List[ConditioningInputType]|None = None) -> List[Image.Image]:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = GenerationParameters(
            safetychecker=False,
            width=size[0],
            height=size[1],
            models=models,
            loras=loras,
            prompt=prompt1,
            negprompt=negprompt,
            steps=steps,
            cfgscale=cfgscale,
            seed=seed1,
            scheduler=scheduler,
            controlimages=conditioning_inputs if conditioning_inputs is not None else []
        )

        if(seed1 is None):
            seed1 = random.randint(0, self.MAX_SEED)
        if(seed2 is None):
            seed2 = random.randint(0, self.MAX_SEED)

        pipelineWrapper = DiffusersPipelines.pipelines.createPipeline(params)
        prompt1 = DiffusersPipelines.pipelines.processPrompt(prompt1, pipelineWrapper)
        prompt2 = DiffusersPipelines.pipelines.processPrompt(prompt2, pipelineWrapper)
        be = BlendingEngine(pipelineWrapper.pipeline)
        be.set_prompt1(prompt1)
        be.set_prompt2(prompt2)
        be.set_negative_prompt(negprompt)
        frames = be.run_transition(fixed_seeds = [seed1, seed2])
        return frames
        