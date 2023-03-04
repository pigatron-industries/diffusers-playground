import torch
import random
import os
import sys
from typing import Dict
from diffusers import ( DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, 
                        StableDiffusionImageVariationPipeline, StableDiffusionInstructPix2PixPipeline,
                        ControlNetModel, StableDiffusionControlNetPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, 
                        UNet2DConditionModel, UniPCMultistepScheduler)
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPModel
from .TextEmbedding import TextEmbeddings
from ..DiffusersModelPresets import DiffusersModelList, DiffusersModel
from ..ModelUtils import getModelsDir, downloadModel, convertToDiffusers
from ..FileUtils import getPathsFiles
from ..StringUtils import mergeDicts

DEFAULT_AUTOENCODER_MODEL = 'stabilityai/sd-vae-ft-mse'
DEFAULT_TEXTTOIMAGE_MODEL = 'runwayml/stable-diffusion-v1-5'
DEFAULT_INPAINT_MODEL = 'runwayml/stable-diffusion-inpainting'
DEFAULT_CONTROLNET_MODEL = 'takuma104/control_sd15_canny'
DEFAULT_DEPTHTOIMAGE_MODEL = 'stabilityai/stable-diffusion-2-depth'
DEFAULT_IMAGEVARIATION_MODEL = 'lambdalabs/sd-image-variations-diffusers'
DEFAULT_INSTRUCTPIXTOPIX_MODEL = 'timbrooks/instruct-pix2pix'
DEFAULT_UPSCALE_MODEL = 'stabilityai/stable-diffusion-x4-upscaler'
DEFAULT_CLIP_MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
DEFAULT_DEVICE = 'cuda'
MAX_SEED = 4294967295


# Use to bypass safety checker as some pipelines don't like None
def dummy(images, **kwargs):
    return images, False

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


class DiffusersPipeline:
    def __init__(self, preset:DiffusersModel, pipeline:DiffusionPipeline, controlmodel:str = None):
        self.preset = preset
        self.pipeline = pipeline
        self.controlmodel = controlmodel


class BaseModelData:
    def __init__(self, base : str, textembeddings : TextEmbeddings, modifierdict = None):  #: Dict[str, list[str]]
        self.base : str = base
        self.textembeddings : TextEmbeddings = textembeddings
        if (modifierdict is None):
            self.modifierdict = {}
        else:
            self.modifierdict = modifierdict


class DiffusersPipelines:

    def __init__(self, localmodelpath = '', device = DEFAULT_DEVICE, safety_checker = True, common_modifierdict = None, custom_pipeline = None):
        self.localmodelpath: str = localmodelpath
        self.localmodelcache: str = getModelsDir()
        self.device: str = device
        self.inferencedevice: str = 'cpu' if self.device == 'mps' else self.device
        self.safety_checker: bool = safety_checker
        if (common_modifierdict is None):
            self.common_modifierdict = {}
        else:
            self.common_modifierdict = common_modifierdict
        self.custom_pipeline = custom_pipeline;

        self.pipelines: Dict[str,DiffusersPipeline] = {}

        self.vae = None
        self.baseModelData: Dict[str, BaseModelData] = {}

        self.presetsImage: DiffusersModelList = DiffusersModelList()
        self.presetsInpaint: DiffusersModelList = DiffusersModelList()
        self.presetsControl: DiffusersModelList = DiffusersModelList()
        self.presetsMisc: DiffusersModelList = DiffusersModelList()

    def loadPresetFile(self, filepath):
        self.presetsImage = DiffusersModelList.from_file(filepath, 'image')
        self.presetsInpaint = DiffusersModelList.from_file(filepath, 'inpaint')
        self.presetsControl = DiffusersModelList.from_file(filepath, 'controlnet')
        self.presetsMisc = DiffusersModelList.from_file(filepath, 'misc')

    def addPresetsImage(self, presets):
        self.presetsImage.addModels(presets)

    def addPresetsInpaint(self, presets):
        self.presetsInpaint.addModels(presets)

    def addPresetsControl(self, presets):
        self.presetsControl.addModels(presets)

    def addPresetImage(self, modelid, base, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        self.presetsImage.addModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)

    def addPresetInpaint(self, modelid, base, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        self.presetsInpaint.addModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)

    def addPresetControl(self, modelid, base, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        self.presetsControl.addModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)


    def getModifierDict(self, base):
        modifiers = self.common_modifierdict.copy()
        modifiers.update(self.baseModelData[base].modifierdict)
        return modifiers


    def loadAutoencoder(self, model = DEFAULT_AUTOENCODER_MODEL):
        self.vae = AutoencoderKL.from_pretrained(model)
            

    def loadTextEmbeddings(self, directory):
        for path, base in getPathsFiles(f"{directory}/*/"):
            if base not in self.baseModelData:
                self.baseModelData[base] = BaseModelData(base, TextEmbeddings(base))
            self.baseModelData[base].textembeddings.load_directory(path, base)
            self.baseModelData[base].modifierdict = self.baseModelData[base].textembeddings.modifiers


    def loadTextEmbedding(self, path, base, token=None):
        if base not in self.baseModelData:
            self.baseModelData[base] = BaseModelData(base, TextEmbeddings(base))
        self.baseModelData[base].textembeddings.load_file(path, token)


    def addTextEmbeddingsToPipeline(self, pipeline: DiffusersPipeline):
        if (pipeline.preset.base in self.baseModelData):
            self.baseModelData[pipeline.preset.base].textembeddings.add_to_model(pipeline.pipeline.text_encoder, pipeline.pipeline.tokenizer)


    def processPrompt(self, prompt: str, pipeline: DiffusersPipeline):
        """ expands embedding tokens into multiple tokens, for each vector in embedding """
        if (pipeline.preset.base in self.baseModelData):
            prompt = self.baseModelData[pipeline.preset.base].textembeddings.process_prompt(prompt)
        return prompt


    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model)
        self.clip_model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16)


    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed


    def loadScheduler(self, schedulerClass, pipeline: DiffusersPipeline):
        if (isinstance(schedulerClass, str)):
            schedulerClass = str_to_class(schedulerClass)
        pipeline.pipeline.scheduler = schedulerClass.from_config(pipeline.pipeline.scheduler.config)


    def getModel(self, modelid, modelList:DiffusersModelList):
        preset = modelList.getModel(modelid)
        if(preset.location == 'url' and preset.modelpath.startswith('http')):
            localmodelpath = self.localmodelcache + '/' + modelid
            if (not os.path.isdir(localmodelpath)):
                downloadModel(preset.modelpath, modelid)
                convertToDiffusers(modelid)
            preset.modelpath = localmodelpath
        return preset


    def latentsToImage(self, pipeline, latents):
        latents = 1 / 0.18215 * latents
        image = pipeline.vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        return pipeline.numpy_to_pil(image)


    #=============== LOAD PIPELINES ==============

    def createArgs(self, preset):
        args = {}
        args['torch_dtype'] = torch.float16
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.revision is not None):
            args['revision'] = preset.revision
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        return args

    def createPipeline(self, cls, model, presets, default, **kwargs):
        if(cls.__name__ not in self.pipelines and (model is None or model == "")):
            model = default
        if(model is None or model == ""):
            return self.pipelines[cls.__name__]
        if (cls.__name__ in self.pipelines and self.pipelines[cls.__name__].preset.modelid == model):
            return self.pipelines[cls.__name__]
        print(f"Creating {cls.__name__} pipeline from model {model}")
        torch.cuda.empty_cache()
        preset = self.getModel(model, presets)
        args = self.createArgs(preset)
        args = mergeDicts(args, kwargs)
        pipeline = cls.from_pretrained(preset.modelpath, **args)
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        self.pipelines[cls.__name__] = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelines[cls.__name__])
        return self.pipelines[cls.__name__]
    
    def createControlNetPipeline(self, model, presets, default, controlmodel, **kwargs):
        if("StableDiffusionControlNetPipeline" not in self.pipelines and (model is None or model == "")):
            model = default
        if(model is None or model == ""):
            return self.pipelines["StableDiffusionControlNetPipeline"]
        if("StableDiffusionControlNetPipeline" in self.pipelines and 
           self.pipelines["StableDiffusionControlNetPipeline"].preset.modelid == model and 
           self.pipelines["StableDiffusionControlNetPipeline"].controlmodel == controlmodel):
            return self.pipelines["StableDiffusionControlNetPipeline"]
        if("StableDiffusionControlNetPipeline" in self.pipelines):
            del self.pipelines["StableDiffusionControlNetPipeline"]
        controlnet = ControlNetModel.from_pretrained(controlmodel, torch_dtype=torch.float16)
        pipeline = self.createPipeline(StableDiffusionControlNetPipeline, model, presets, default, controlnet=controlnet)
        pipeline.controlmodel = controlmodel
        return pipeline


    #=============== INFERENCE ==============

    def inference(self, pipeline:DiffusersPipeline, prompt, seed, scheduler=None, tiling=False, **kwargs):
        prompt = self.processPrompt(prompt, pipeline)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, pipeline)
        pipeline.pipeline.vae.enable_tiling(tiling)
        if(pipeline.preset.autocast):
            with torch.autocast(self.inferencedevice):
                image = pipeline.pipeline(prompt, generator=generator, **kwargs).images[0]
        else:
            image = pipeline.pipeline(prompt, generator=generator, **kwargs).images[0]
        return image, seed


    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None, scheduler=None, model=None, tiling=False, **kwargs):
        pipeline = self.createPipeline(StableDiffusionPipeline, model, self.presetsImage, DEFAULT_TEXTTOIMAGE_MODEL, custom_pipeline=self.custom_pipeline)
        return self.inference(prompt=prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, 
                              width=width, height=height, pipeline=pipeline, seed=seed, scheduler=scheduler, tiling=tiling)


    def imageToImage(self, initimage, prompt, negprompt, strength, scale, seed=None, scheduler=None, model=None, tiling=False, **kwargs):        
        pipeline = self.createPipeline(StableDiffusionImg2ImgPipeline, model, self.presetsImage, DEFAULT_TEXTTOIMAGE_MODEL)
        initimage = initimage.convert("RGB")
        return self.inference(prompt=prompt, image=initimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, 
                              pipeline=pipeline, seed=seed, scheduler=scheduler, tiling=tiling)


    def controlNet(self, initimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, model=None, controlmodel=None, tiling=False, **kwargs):
        pipeline = self.createControlNetPipeline(model, self.presetsImage, DEFAULT_TEXTTOIMAGE_MODEL, controlmodel)
        initimage = initimage.convert("RGB")
        return self.inference(prompt=prompt, image=initimage, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, 
                              pipeline=pipeline, seed=seed, scheduler=scheduler, tiling=tiling)


    def inpaint(self, initimage, maskimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, model=None, tiling=False, **kwargs):
        pipeline = self.createPipeline(StableDiffusionInpaintPipeline, model, self.presetsInpaint, DEFAULT_INPAINT_MODEL)
        initimage = initimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        return self.inference(prompt=prompt, image=initimage, mask_image=maskimage, width=initimage.width, height=initimage.height,
                              negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, pipeline=pipeline, seed=seed, 
                              scheduler=scheduler, tiling=tiling)


    def depthToImage(self, inimage, prompt, negprompt, strength, scale, steps=50, seed=None, scheduler=None, model=None, **kwargs):
        pipeline = self.createPipeline(StableDiffusionDepth2ImgPipeline, model, self.presetsImage, DEFAULT_DEPTHTOIMAGE_MODEL)
        inimage = inimage.convert("RGB")
        return self.inference(prompt=prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, 
                              num_inference_steps=steps, pipeline=pipeline, seed=seed, scheduler=scheduler)


    def imageVariation(self, initimage, steps, scale, seed=None, scheduler=None, model=None, **kwargs):
        pipeline = self.createPipeline(StableDiffusionImageVariationPipeline, model, self.presetsImage, DEFAULT_IMAGEVARIATION_MODEL)
        return self.inference(image=initimage.convert("RGB"), width=initimage.width, height=initimage.height, num_inference_steps=steps, 
                              guidance_scale=scale, pipeline=pipeline, seed=seed, scheduler=scheduler)


    def instructPixToPix(self, initimage, prompt, steps, scale, seed=None, scheduler=None, model=None, **kwargs):
        pipeline = self.createPipeline(StableDiffusionInstructPix2PixPipeline, model, self.presetsImage, DEFAULT_INSTRUCTPIXTOPIX_MODEL)
        return self.inference(prompt, image=initimage.convert("RGB"), num_inference_steps=steps, guidance_scale=scale, pipeline=pipeline, seed=seed, scheduler=scheduler)


    def upscale(self, inimage, prompt, scheduler=None, model=None):
        pipeline = self.createPipeline(StableDiffusionUpscalePipeline, model, self.presetsImage, DEFAULT_UPSCALE_MODEL)
        return self.inference(image=inimage, prompt=prompt, pipeline=pipeline, scheduler=scheduler)