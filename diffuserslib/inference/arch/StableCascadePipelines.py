from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline


class StableCascadePipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.createPipeline(params)
        super().__init__(params, inferencedevice)


    def createPipeline(self, params:GenerationParameters):
        if(params.modelConfig is None or params.modelConfig.data is None):
            raise ValueError("Must provide modelConfig")
        pipeline_params = self.createPipelineParams(params)
        self.pipeline_prior = StableCascadePriorPipeline.from_pretrained(params.modelConfig.data['prior'], **pipeline_params).to(self.device)
        self.pipeline_decoder = StableCascadeDecoderPipeline.from_pretrained(params.modelConfig.data['decoder'], **pipeline_params).to(self.device)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    
    
    def diffusers_inference(self, prompt, negative_prompt, seed, guidance_scale=4.0, **kwargs):
        generator, seed = self.createGenerator(seed)
        prior_output = self.pipeline_prior(prompt=prompt, negative_prompt=negative_prompt, generator=generator, guidance_scale=guidance_scale)
        image = self.pipeline_decoder(image_embeddings=prior_output.image_embeddings.half(), prompt=prompt, negative_prompt=negative_prompt, guidance_scale=0.0, output_type="pil", **kwargs).images[0]
        return image, seed


class StableCascadeGeneratePipelineWrapper(StableCascadePipelineWrapper):

    PIPELINE_MAP = {
        #img2img,   inpaint
        (False,     False):    StableCascadePriorPipeline,
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
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height
    
    
    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addCommonParams(params, diffusers_params)
        return super().diffusers_inference(**diffusers_params)

