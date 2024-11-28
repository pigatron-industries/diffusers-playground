from .StableDiffusionPipelines import DiffusersPipelineWrapper
from ..GenerationParameters import GenerationParameters, ControlImageType
from diffuserslib.models.DiffusersModelPresets import DiffusersModelType
from diffuserslib.ModelUtils import getFile
from diffuserslib.models.DiffusersModelPresets import DiffusersModel
from PIL import  ImageOps
from diffusers import ( # Schedulers
                        FlowMatchEulerDiscreteScheduler)
from safetensors import safe_open
from diffuserslib.scripts.convert_flux_lora import convert_sd_scripts_to_ai_toolkit


class FluxPipelineWrapper(DiffusersPipelineWrapper):
    def __init__(self, cls, params:GenerationParameters, device, **kwargs):
        from diffusers import FluxControlNetModel
        self.safety_checker = params.safetychecker
        self.device = device
        inferencedevice = 'cpu' if self.device == 'mps' else self.device
        super().__init__(params, inferencedevice, cls, controlnet_cls = FluxControlNetModel, **kwargs)


    def loadPipeline(self, modelConfig:DiffusersModel, cls, pipelineParams):
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        if (modelConfig.modelpath.endswith('.safetensors') or modelConfig.modelpath.endswith('.ckpt')):
            hf_file = getFile(modelConfig.modelpath)
            transformer = FluxTransformer2DModel.from_single_file(hf_file, **pipelineParams).to(self.device)
            if(modelConfig.base == 'flux_1_s'):
                baseModel = 'black-forest-labs/FLUX.1-schnell'
            else:
                baseModel = 'black-forest-labs/FLUX.1-dev'
            pipeline = cls.from_pretrained(baseModel, transformer=None, **pipelineParams).to(self.device)
            pipeline.transformer = transformer
            return pipeline
        else:
            return cls.from_pretrained(modelConfig.modelpath, **pipelineParams).to(self.device)


    def createPipelineParams(self, params:GenerationParameters):
        pipeline_params = {}
        self.addPipelineParamsCommon(params, pipeline_params)
        if(self.features.controlnet):
            self.addPipelineParamsControlNet(params, pipeline_params)
        return pipeline_params
    

    def addPipelineParamsControlNet(self, params:GenerationParameters, pipeline_params):
        args = {}
        if(self.dtype is not None):
            args['torch_dtype'] = self.dtype
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        controlnet = []
        for controlnetparam in controlnetparams:
            controlnet.append(self.controlnet_cls.from_pretrained(controlnetparam.model, **args))
        pipeline_params['controlnet'] = controlnet[0] #Only one controlnet model is supported by flus pipeline currently
        return pipeline_params
    
    
    def diffusers_inference(self, prompt, seed, guidance_scale=4.0, scheduler=None, negative_prompt=None, clip_skip=None, **kwargs):
        generator, seed = self.createGenerator(seed)
        output = self.pipeline(prompt=prompt, generator=generator, guidance_scale=guidance_scale, return_dict=True, **kwargs)
        return output, seed
    

    def load_lora_weights(self, path:str):
        state_dict = {}
        sdscripts_format = False
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            for k in f.keys():
                if(k.startswith('lora_unet')):
                    sdscripts_format = True
                state_dict[k] = f.get_tensor(k)
        if(sdscripts_format):
            state_dict = convert_sd_scripts_to_ai_toolkit(state_dict)
        return state_dict
        

    def add_lora(self, lora):
        if(lora.name not in self.lora_names):
            state_dict = self.load_lora_weights(lora.path)
            self.lora_names.append(lora.name)
            self.pipeline.load_lora_weights(state_dict, adapter_name=lora.name.split('.', 1)[0])


class FluxGeneratePipelineWrapper(FluxPipelineWrapper):

    def __init__(self, params:GenerationParameters, device):
        cls = self.getPipelineClass(params)
        super().__init__(params=params, device=device, cls=cls)


    def getPipelineClass(self, params:GenerationParameters):
        from diffusers import FluxControlNetPipeline, FluxImg2ImgPipeline, FluxFillPipeline, FluxPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetInpaintPipeline
        PIPELINE_MAP = {
            #img2img,   inpaint, controlnet
            (False,     False,   False):    FluxPipeline,
            (True,      False,   False):    FluxImg2ImgPipeline,
            (True,      True,    False):    FluxFillPipeline,
            (False,     False,   True):     FluxControlNetPipeline,
            (True,      False,   True):     FluxControlNetImg2ImgPipeline,
            (False,     True,    False):    FluxControlNetInpaintPipeline,
        }
        self.features = self.getPipelineFeatures(params)
        if(self.features.differential):
            return "pipeline_flux_differential_img2img"
        return PIPELINE_MAP[(self.features.img2img, self.features.inpaint, self.features.controlnet)]


    def addInferenceParamsImg2Img(self, params:GenerationParameters, diffusers_params):
        #FluxImg2ImgPipeline not using dimnsions of image
        initimageparams = params.getInitImage()
        if(initimageparams is not None and initimageparams.image is not None):
            diffusers_params['image'] = initimageparams.image.convert("RGB")
            diffusers_params['strength'] = initimageparams.condscale
            diffusers_params['width'] = initimageparams.image.width
            diffusers_params['height'] = initimageparams.image.height


    def addInferenceParamsInpaint(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getMaskImage()
        if(initimageparams is None or maskimageparams is None or initimageparams.image is None or maskimageparams.image is None):
            raise ValueError("Must provide both initimage and maskimage")
        diffusers_params['image'] = initimageparams.image.convert("RGB")
        diffusers_params['mask_image'] = maskimageparams.image.convert("RGB")
        diffusers_params['num_inference_steps'] = params.steps
        diffusers_params['width'] = initimageparams.image.width
        diffusers_params['height'] = initimageparams.image.height
        del diffusers_params['strength']


    def addInferenceParamsDifferential(self, params:GenerationParameters, diffusers_params):
        initimageparams = params.getInitImage()
        maskimageparams = params.getImage(ControlImageType.IMAGETYPE_DIFFMASKIMAGE)
        if(initimageparams is not None and maskimageparams is not None):
            diffusers_params['image'] = initimageparams.image.convert("RGB")
            diffusers_params['mask_image'] = ImageOps.invert(maskimageparams.image.convert("L"))


    def addInferenceParamsControlNet(self, params:GenerationParameters, diffusers_params):
        controlnetparams = params.getConditioningParamsByModelType(DiffusersModelType.controlnet)
        images = []
        scales = []
        for controlnetparam in controlnetparams:
            if(controlnetparam.image is not None and controlnetparam.condscale > 0):
                images.append(controlnetparam.image.convert("RGB"))
                scales.append(controlnetparam.condscale)
        diffusers_params['control_image'] = images[0] #Only one controlnet model is supported by flus pipeline currently
        diffusers_params['controlnet_conditioning_scale'] = scales[0]