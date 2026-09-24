from .DiffusersPipelineWrapper import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters
from diffusers import Flux2Pipeline


class Flux2PipelineWrapper(DiffusersPipelineWrapper):
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
        # Flux2 is a flow-matching model and does not support negative prompts
        generator, seed = self.createGenerator(seed)
        prompt = "test"
        output = self.pipeline(prompt=prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True)
        return output, seed


class Flux2GeneratePipelineWrapper(Flux2PipelineWrapper):

    PIPELINE_MAP = {
        #img2img,  inpaint
        (False,     False):    Flux2Pipeline,
        (True,      False):    Flux2Pipeline,
    }

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)

    def getPipelineClass(self, params:GenerationParameters):
        self.features = self.getPipelineFeatures(params)
        return self.PIPELINE_MAP[(self.features.img2img, self.features.inpaint)]


    def addInferenceParamsImg2Img(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        if(initimageparams is not None and initimageparams.image is not None):
            # Flux2Pipeline takes reference images (single image or list) and has no strength parameter
            diffusers_params['image'] = initimageparams.image.convert("RGB")
            diffusers_params['width'] = initimageparams.image.width
            diffusers_params['height'] = initimageparams.image.height
            diffusers_params['num_inference_steps'] = params.steps
