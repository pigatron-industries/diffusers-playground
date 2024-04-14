from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters, ModelParameters, LoraParameters, ControlImageParameters, ControlImageType
from PIL import Image

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]
LorasType = List[LoraParameters]
LorasFuncType = LorasType | Callable[[], LorasType]

class VideoDiffusionPIANode(FunctionalNode):

    def __init__(self,
                 image:ImageFuncType,
                 models:ModelsFuncType = [],
                 loras:LorasFuncType = [],
                 size:SizeFuncType = (512, 512),
                 prompt:StringFuncType = "",
                 negprompt:StringFuncType = "",
                 steps:IntFuncType = 40,
                 strength:FloatFuncType = 1.0,
                 cfgscale:FloatFuncType = 7.0,
                 seed:IntFuncType|None = None,
                 scheduler:StringFuncType = "DPMSolverMultistepScheduler",
                 frames:IntFuncType = 16,
                 name:str = "image_diffusion"):
        super().__init__(name)
        self.addParam("image", image, Image.Image)
        self.addParam("size", size, SizeType)
        self.addParam("models", models, ModelsType)
        self.addParam("loras", loras, LorasType)
        self.addParam("prompt", prompt, str)
        self.addParam("negprompt", negprompt, str)
        self.addParam("steps", steps, int)
        self.addParam("strength", strength, float)
        self.addParam("cfgscale", cfgscale, float)
        self.addParam("seed", seed, int)
        self.addParam("scheduler", scheduler, str)
        self.addParam("frames", frames, int)


    def process(self, 
                image:Image.Image,
                size:SizeType, 
                models:ModelsType, 
                loras:LorasType,
                prompt:str, 
                negprompt:str, 
                steps:int, 
                strength:float,
                cfgscale:float, 
                seed:int|None, 
                scheduler:str,
                frames:int) -> List[Image.Image]:
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        
        params = GenerationParameters(
            generationtype="pia",
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
            frames=frames,
            controlimages=[ControlImageParameters(image = image, condscale = strength, type = ControlImageType.IMAGETYPE_INITIMAGE, model = "initimage")]
        )

        output, seed = DiffusersPipelines.pipelines.generate(params)
        # output = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        if(isinstance(output, List)):
            return output
        else:
            raise Exception("Output is not a list of images")
        