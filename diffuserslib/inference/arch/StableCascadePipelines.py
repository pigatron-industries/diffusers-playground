from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from ...StringUtils import mergeDicts
from ...models.DiffusersModelPresets import DiffusersModel
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline


class StableCascadePipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, params:GenerationParameters, device, **kwargs):
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls=None, **kwargs)


    def createPipeline(self, params:GenerationParameters):
        if(params.modelConfig is None or params.modelConfig.data is None):
            raise ValueError("Must provide modelConfig")
        pipeline_params = self.createPipelineParams(params)
        self.pipeline_prior = StableCascadePriorPipeline.from_pretrained(params.modelConfig.data['prior'], variant="bf16", **pipeline_params).to(self.device)
        self.pipeline_decoder = StableCascadeDecoderPipeline.from_pretrained(params.modelConfig.data['decoder'], variant="bf16", **pipeline_params).to(self.device)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        return pipeline_params
    
    
    def diffusers_inference(self, prompt, negative_prompt, width, height, seed, prior_guidance_scale, prior_num_inference_steps, decoder_guidance_scale, images = None, **kwargs):
        generator, seed = self.createGenerator(seed)
        prior_output = self.pipeline_prior(prompt=prompt, negative_prompt=negative_prompt, images = images,
                                           generator=generator, guidance_scale=prior_guidance_scale, num_inference_steps=prior_num_inference_steps, 
                                           width=width, height=height)
        image = self.pipeline_decoder(image_embeddings=prior_output.image_embeddings, prompt=prompt, negative_prompt=negative_prompt, 
                                      guidance_scale=decoder_guidance_scale,
                                      output_type="pil", **kwargs).images[0]
        return image, seed


class StableCascadeGeneratePipelineWrapper(StableCascadePipelineWrapper):

    PIPELINE_MAP = {
        #img2img,   inpaint
        (False,     False):    StableCascadePriorPipeline,
        (True,      False):    StableCascadePriorPipeline,
    }

    def __init__(self, params:GenerationParameters, device):
        self.features = self.getPipelineFeatures(params)
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]


    def addInferenceParamsCommon(self, params:GenerationParameters, diffusers_params):
        diffusers_params['prompt'] = params.prompt
        diffusers_params['negative_prompt'] = params.negprompt
        diffusers_params['seed'] = params.seed
        diffusers_params['prior_guidance_scale'] = params.cfgscale
        diffusers_params['decoder_guidance_scale'] = 0.0
        diffusers_params['prior_num_inference_steps'] = params.steps
        diffusers_params['num_inference_steps'] = params.steps
        


    def addInferenceParamsTxt2Img(self, params:GenerationParameters, diffusers_params):
        diffusers_params['width'] = params.width
        diffusers_params['height'] = params.height

    def addInferenceParamsImg2Img(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        if(initimageparams is not None and initimageparams.image is not None):
            diffusers_params['images'] = [initimageparams.image.convert("RGB")]
            diffusers_params['width'] = initimageparams.image.width
            diffusers_params['height'] = initimageparams.image.height
    
    
    def inference(self, params:GenerationParameters):
        diffusers_params = {}
        self.addInferenceParamsCommon(params, diffusers_params)
        if(not self.features.img2img):
            self.addInferenceParamsTxt2Img(params, diffusers_params)
        if(self.features.img2img):
            self.addInferenceParamsImg2Img(params, diffusers_params)
        return super().diffusers_inference(**diffusers_params)

