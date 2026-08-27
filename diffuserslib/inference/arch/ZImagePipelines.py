from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImageTransformer2DModel
import torch


class ZImagePipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, **kwargs)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params

    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, scheduler=None, clip_skip=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed

    def loadPipeline(self, modelConfig, cls, pipelineParams):
        print("Load ZImage checkpoint: ", modelConfig.modelpath)
        if (modelConfig.modelpath.endswith('.safetensors') or modelConfig.modelpath.endswith('.ckpt')):
            transformer = ZImageTransformer2DModel.from_single_file(modelConfig.modelpath, torch_dtype=torch.bfloat16)
            # transformer.set_attention_backend("native")
            pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=torch.bfloat16).to(self.device)
            pipe.vae.to(torch.float32) 
            return pipe
        else:
            return cls.from_pretrained(modelConfig.modelpath, **pipelineParams).to(self.device)


class ZImageGeneratePipelineWrapper(ZImagePipelineWrapper):

    PIPELINE_MAP = {
        #img2img,  inpaint
        (False,     False):    ZImagePipeline,
        (True,      False):    ZImageImg2ImgPipeline,
        (True,      True):     ZImageInpaintPipeline,
    }

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)

    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]
