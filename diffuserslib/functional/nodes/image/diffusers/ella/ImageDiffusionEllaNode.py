from diffuserslib.functional.FunctionalNode import FunctionalNode
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.inference.DiffusersPipelines import DiffusersPipelines
from diffuserslib.inference.GenerationParameters import GenerationParameters, ModelParameters, LoraParameters
from diffuserslib.inference.arch.StableDiffusionPipelines import StableDiffusionGeneratePipelineWrapper
from ..ConditioningInputNode import ConditioningInputType, ConditioningInputFuncsType
from .EllaInference import load_ella, ELLAProxyUNet
from .EllaModel import T5TextEmbedder
from PIL import Image
import huggingface_hub
import torch

ModelsType = List[ModelParameters]
ModelsFuncType = ModelsType | Callable[[], ModelsType]
LorasType = List[LoraParameters]
LorasFuncType = LorasType | Callable[[], LorasType]

class ImageDiffusionEllaNode(FunctionalNode):
    """ 
    ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment
    https://github.com/TencentQQGYLab/ELLA 
    """

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
        self.addParam("conditioning_inputs", conditioning_inputs, List[ConditioningInputType])
        self.ella = None
        self.t5_encoder = None


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
        
        if(self.ella is None):
            self.loadModels()
        
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
            controlimages=conditioningparams
        )

        pipelineWrapper = DiffusersPipelines.pipelines.createPipeline(params)
        if (not isinstance(pipelineWrapper, StableDiffusionGeneratePipelineWrapper)):
            raise Exception("Pipeline is not a StableDiffusionGeneratePipelineWrapper")
        DiffusersPipelines.pipelines.processPrompt(params, pipelineWrapper)
        pipelineWrapper.pipeline.unet = ELLAProxyUNet(self.ella, pipelineWrapper.pipeline.unet)
        diffusers_params = pipelineWrapper.createInferenceParams(params)
        prompt_embeds = self.t5_encoder(prompt, max_length=128).to(DiffusersPipelines.pipelines.device, torch.float32)
        diffusers_params['prompt_embeds'] = prompt_embeds
        del diffusers_params['prompt']
        negative_prompt_embeds = self.t5_encoder(negprompt, max_length=128).to(DiffusersPipelines.pipelines.device, torch.float32)
        diffusers_params['negative_prompt_embeds'] = negative_prompt_embeds
        del diffusers_params['negative_prompt']
        output, seed = pipelineWrapper.diffusers_inference_embeds(**diffusers_params)

        if(isinstance(output.images[0], Image.Image)):
            return output.images[0]
        else:
            raise Exception("Output is not an image")
        

    def loadModels(self):
        if(DiffusersPipelines.pipelines is None):
            raise Exception("DiffusersPipelines is not initialized")
        path = huggingface_hub.hf_hub_download(repo_id = 'QQGYLab/ELLA', subfolder = None, filename = 'ella-sd1.5-tsc-t5xl.safetensors')
        self.ella = load_ella(path, DiffusersPipelines.pipelines.device, torch.float32)
        self.t5_encoder = T5TextEmbedder().to(DiffusersPipelines.pipelines.device, dtype=torch.float32)