from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import PixArtSigmaPipeline
import torch


class PixartSigmaPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params, cls, **kwargs)
        super().__init__(params, inferencedevice)

    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    
    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, scheduler=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        image = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs).images[0]
        return image, seed


class PixartSigmaGeneratePipelineWrapper(PixartSigmaPipelineWrapper):

    PIPELINE_MAP = {
        #img2im,    inpaint
        (False,     False):    PixArtSigmaPipeline
    }

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)

    def getPipelineClass(self, params:GenerationParameters):
        self.is_img2img = False
        self.is_inpaint = False
        for conditioningimage in params.controlimages:
            if(conditioningimage.type == ControlImageType.IMAGETYPE_INITIMAGE):
                self.is_img2img = True
            if(conditioningimage.type == ControlImageType.IMAGETYPE_MASKIMAGE):
                self.is_inpaint = True
        return self.PIPELINE_MAP[(self.is_img2img, self.is_inpaint)]


    def addCommonParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['guidance_scale'] = params.cfgscale
        diffusers_params['scheduler'] = params.scheduler
    
    def addImg2ImgParams(self, params:GenerationParameters, diffusers_params):
        initimage = params.getInitImage()
        if(initimage is not None and initimage.image is not None):
            diffusers_params['image'] = initimage.image.convert("RGB")
            diffusers_params['strength'] = params.strength

    def addTxt2ImgParams(self, params:GenerationParameters, diffusers_params):
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
        diffusers_params['num_inference_steps'] = params.steps

    def addInpaintParams(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None or initimageparams.image is None or maskimageparams.image is None):
            raise ValueError("Must provide both initimage and maskimage")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['mask_image'] = maskimageparams.image.convert("RGB")
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['strength'] = params.strength
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height
    
    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addCommonParams(params, diffusers_params)
        if(not self.is_img2img):
            self.addTxt2ImgParams(params, diffusers_params)
        if(self.is_img2img):
            self.addImg2ImgParams(params, diffusers_params)
        if(self.is_inpaint):
            self.addInpaintParams(params, diffusers_params)
        return super().diffusers_inference(**diffusers_params)

