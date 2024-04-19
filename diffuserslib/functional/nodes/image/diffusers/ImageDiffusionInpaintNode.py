from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters, ControlImageType, ModelParameters, LoraParameters
from diffuserslib.ImageUtils import alphaToMask, compositeImages
from .ImageDiffusionNode import ModelsFuncType, LorasFuncType, ModelsType, LorasType
from .ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from PIL import Image


class ImageDiffusionInpaintNode(FunctionalNode):

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
                 conditioning_inputs:ConditioningInputFuncsType|None = None,        
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
        self.addParam("conditioning_inputs", conditioning_inputs, List[ConditioningInputType])


    def process(self, image:Image.Image,conditioning_inputs:List[ConditioningInputType]|None, **kwargs) -> Image.Image:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        conditioningparams = []
        if(conditioning_inputs is not None):
            for conditioning_input in conditioning_inputs:
                if(conditioning_input.image is not None):
                    conditioningparams.append(conditioning_input)

        inpaint_conditioningparams = conditioningparams.copy()
        initimage = image.convert("RGB")
        maskimage = alphaToMask(image)
        inpaint_conditioningparams.append(ConditioningInputType(image=initimage, type=ControlImageType.IMAGETYPE_INITIMAGE))
        inpaint_conditioningparams.append(ConditioningInputType(image=maskimage, type=ControlImageType.IMAGETYPE_MASKIMAGE))
        params = GenerationParameters(safetychecker=False, controlimages=conditioningparams, **kwargs)
        inpaintedimage, _ = DiffusersPipelines.pipelines.generate(params)
        assert isinstance(inpaintedimage, Image.Image)
        
        outimage = compositeImages(inpaintedimage, initimage, maskimage, maskDilation=21, maskFeather=3)
        return outimage
        


        