import torch
import random
import os
import sys
from typing import Dict
from diffusers import ( DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, 
                        StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, 
                        StableDiffusionImageVariationPipeline, StableDiffusionInstructPix2PixPipeline,
                        # StableDiffusionControlNetPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, 
                        UNet2DConditionModel)
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPModel
from .TextEmbedding import TextEmbeddings
from ..DiffusersModelPresets import DiffusersModelList, DiffusersModel
from ..ModelUtils import getModelsDir, downloadModel, convertToDiffusers
from ..FileUtils import getPathsFiles

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
    def __init__(self, preset:DiffusersModel, pipeline:DiffusionPipeline):
        self.preset = preset
        self.pipeline = pipeline


class BaseModelData:
    def __init__(self, base : str, textembeddings : TextEmbeddings, modifierdict = None):  #: Dict[str, list[str]]
        self.base : str = base
        self.textembeddings : TextEmbeddings = textembeddings
        if (modifierdict is None):
            self.modifierdict = {}
        else:
            self.modifierdict = modifierdict


class DiffusersPipelines:

    def __init__(self, localmodelpath = '', device = DEFAULT_DEVICE, safety_checker = True, common_modifierdict = None):
        self.localmodelpath: str = localmodelpath
        self.localmodelcache: str = getModelsDir()
        self.device: str = device
        self.inferencedevice: str = 'cpu' if self.device == 'mps' else self.device
        self.safety_checker: bool = safety_checker
        if (common_modifierdict is None):
            self.common_modifierdict = {}
        else:
            self.common_modifierdict = common_modifierdict

        self.pipelineTextToImage: DiffusersPipeline = None
        self.pipelineImageToImage: DiffusersPipeline = None
        self.pipelineInpainting: DiffusersPipeline = None
        self.pipelineControlNet: DiffusersPipeline = None
        self.pipelineDepthToImage: DiffusersPipeline = None
        self.pipelineImageVariation: DiffusersPipeline = None
        self.pipelineInstructPixToPix: DiffusersPipeline = None
        self.pipelineUpscale: DiffusersPipeline = None

        self.vae = None
        self.baseModelData: Dict[str, BaseModelData] = {}

        self.presetsImage: DiffusersModelList = DiffusersModelList()
        self.presetsInpaint: DiffusersModelList = DiffusersModelList()
        self.presetsControl: DiffusersModelList = DiffusersModelList()
        self.presetsMisc: DiffusersModelList = DiffusersModelList()

    def loadPresetFile(self, filepath):
        self.presetsImage = DiffusersModelList.from_file(filepath, 'image')
        self.presetsImage = DiffusersModelList.from_file(filepath, 'inpaint')
        self.presetsImage = DiffusersModelList.from_file(filepath, 'controlnet')
        self.presetsImage = DiffusersModelList.from_file(filepath, 'misc')

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

    def createTextToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL, custom_pipeline=None):
        if(self.pipelineTextToImage is not None and self.pipelineTextToImage.preset.modelid == model):
            return
        print(f"Creating text to image pipeline from model {model}")
        self.pipelineTextToImage = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsImage)
        args = self.createArgs(preset)
        if(custom_pipeline is not None and custom_pipeline != ''):
            args['custom_pipeline'] = custom_pipeline
        if(custom_pipeline == 'clip_guided_stable_diffusion'):
            args['feature_extractor'] = self.feature_extractor
            args['clip_model'] = self.clip_model
        pipeline = DiffusionPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineTextToImage = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineTextToImage)


    def createImageToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL):
        if(self.pipelineImageToImage is not None and self.pipelineImageToImage.preset.modelid == model):
            return
        print(f"Creating image to image pipeline from model {model}")
        self.pipelineImageToImage = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsImage)
        args = self.createArgs(preset)
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineImageToImage = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineImageToImage)


    def createInpaintPipeline(self, model=DEFAULT_INPAINT_MODEL):
        if(self.pipelineInpainting is not None and self.pipelineInpainting.preset.modelid == model):
            return
        print(f"Creating inpainting pipeline from model {model}")
        self.pipelineInpainting = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsInpaint)
        args = self.createArgs(preset)
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineInpainting = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineInpainting)


    def createControlNetPipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL, controlmodel=DEFAULT_CONTROLNET_MODEL):
        # WORK IN PROGRESS - untested
        if(self.pipelineControlNet is not None and self.pipelineControlNet.preset.modelid == model):
            return
        print(f"Creating control net pipeline from model {model}")
        self.pipelineControlNet = None
        torch.cuda.empty_cache()
        controlpreset = self.getModel(model, self.presetsControl)
        preset = self.getModel(model, self.presetsImage)
        args = self.createArgs(preset)

        controlnet = UNet2DConditionModel.from_pretrained(controlpreset.modelpath, subfolder="controlnet", torch_dtype=torch.float16)
        diffusionpipeline = DiffusionPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline = StableDiffusionControlNetPipeline(vae=diffusionpipeline.vae, 
                                                     text_encoder=diffusionpipeline.text_encoder, 
                                                     tokenizer=diffusionpipeline.tokenizer,
                                                     controlnet=controlnet,
                                                     unet=diffusionpipeline.unet,
                                                     scheduler=diffusionpipeline.scheduler,
                                                     safety_checker=diffusionpipeline.safety_checker,
                                                     feature_extractor=diffusionpipeline.feature_extractor
                   ).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineControlNet = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineControlNet)


    def createDepthToImagePipeline(self, model=DEFAULT_DEPTHTOIMAGE_MODEL):
        if(self.pipelineDepthToImage is not None and self.pipelineDepthToImage.preset.modelid == model):
            return
        print(f"Creating depth to image pipeline from model {model}")
        self.pipelineDepthToImage = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsMisc)
        args = self.createArgs(preset)
        pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineDepthToImage = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineDepthToImage)


    def createImageVariationPipeline(self, model=DEFAULT_IMAGEVARIATION_MODEL):
        if(self.pipelineImageVariation is not None and self.pipelineInstructPixToPix.preset.modelid == model):
            return
        print(f"Creating image variation pipeline from model {model}")
        self.pipelineImageVariation = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsMisc)
        args = self.createArgs(preset)
        pipeline = StableDiffusionImageVariationPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineImageVariation = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineImageVariation)


    def createInstructPixToPixPipeline(self, model=DEFAULT_INSTRUCTPIXTOPIX_MODEL):
        if(self.pipelineInstructPixToPix is not None and self.pipelineInstructPixToPix.preset.modelid == model):
            return
        print(f"Creating instruct pix to pix pipeline from model {model}")
        self.pipelineInstructPixToPix = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsMisc)
        args = self.createArgs(preset)
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(preset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineInstructPixToPix = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineInstructPixToPix)


    def createUpscalePipeline(self, model=DEFAULT_UPSCALE_MODEL):
        if(self.pipelineUpscale is not None and self.pipelineUpscale.preset.modelid == model):
            return
        print(f"Creating upscale pipeline from model {model}")
        self.pipelineUpscale = None
        torch.cuda.empty_cache()
        preset = self.getModel(model, self.presetsMisc)
        args = {}
        args['torch_dtype'] = torch.float16
        if(preset.fp16):
            args['revision'] = 'fp16'
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(self.upscalePreset.modelpath, **args).to(self.device)
        pipeline.enable_attention_slicing()
        self.pipelineUpscale = DiffusersPipeline(preset, pipeline)
        self.addTextEmbeddingsToPipeline(self.pipelineUpscale)


    #=============== INFERENCE ==============

    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None, scheduler=None, **kwargs):
        if (self.pipelineTextToImage is None):
            raise Exception('text to image pipeline not loaded')
        prompt = self.processPrompt(prompt, self.pipelineTextToImage)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineTextToImage)
        if(self.pipelineTextToImage.preset.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.pipelineTextToImage.pipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        else:
            image = self.pipelineTextToImage.pipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        return image, seed


    def imageToImage(self, inimage, prompt, negprompt, strength, scale, seed=None, scheduler=None, **kwargs):
        if (self.pipelineImageToImage is None):
            raise Exception('image to image pipeline not loaded')
        inimage = inimage.convert("RGB")
        prompt = self.processPrompt(prompt, self.pipelineImageToImage)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineImageToImage)
        if(self.pipelineImageToImage.preset.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.pipelineImageToImage.pipeline(prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, generator=generator).images[0]
        else:
            image = self.pipelineImageToImage.pipeline(prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, generator=generator).images[0]
        return image, seed


    def inpaint(self, initimage, maskimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, **kwargs):
        if (self.pipelineInpainting is None):
            raise Exception('inpainting pipeline not loaded')
        prompt = self.processPrompt(prompt, self.pipelineInpainting)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineInpainting)
        with torch.autocast(self.inferencedevice):
            outimage = self.pipelineInpainting.pipeline(prompt, image=initimage.convert("RGB"), mask_image=maskimage.convert("RGB"), width=initimage.width, height=initimage.height,
                                               negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return outimage, seed


    def controlnet(self, initimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, **kwargs):
        # WORK IN PROGRESS
        if (self.pipelineInpainting is None):
            raise Exception('control net pipeline not loaded')
        prompt = self.processPrompt(prompt, self.pipelineInpainting)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineInpainting)
        with torch.autocast(self.inferencedevice):
            outimage = self.pipelineInpainting.pipeline(prompt=prompt, controlnet_hint=initimage.convert("RGB"), negative_prompt=negprompt, 
                                                        num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return outimage, seed


    def depthToImage(self, inimage, prompt, negprompt, strength, scale, steps=50, seed=None, scheduler=None, **kwargs):
        if (self.pipelineDepthToImage is None):
            raise Exception('depth to image pipeline not loaded')
        inimage = inimage.convert("RGB")
        prompt = self.processPrompt(prompt, self.pipelineDepthToImage)
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineDepthToImage)
        with torch.autocast(self.inferencedevice):
            image = self.pipelineDepthToImage.pipeline(prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, num_inference_steps=steps, generator=generator).images[0]
        return image, seed


    def imageVariation(self, initimage, steps, scale, seed=None, scheduler=None, **kwargs):
        if (self.pipelineImageVariation is None):
            raise Exception('image variation pipeline not loaded')
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineImageVariation)
        with torch.autocast(self.inferencedevice):
            outimage = self.pipelineImageVariation.pipeline(image=initimage.convert("RGB"), width=initimage.width, height=initimage.height, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return outimage, seed


    def instructPixToPix(self, initimage, prompt, steps, scale, seed=None, scheduler=None, **kwargs):
        if (self.pipelineInstructPixToPix is None):
            raise Exception('image instruct pix to pix not loaded')
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineInstructPixToPix)
        with torch.autocast(self.inferencedevice):
            outimage = self.pipelineInstructPixToPix.pipeline(prompt, image=initimage.convert("RGB"), num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return outimage, seed


    def upscale(self, inimage, prompt, scheduler=None, model=None):
        if (self.pipelineUpscale is None and model is None):
            model = DEFAULT_UPSCALE_MODEL
        if(model is not None and model != ""):
            print(f'Model: {model}')
            self.createUpscalePipeline(model)
        prompt = self.processPrompt(prompt, self.pipelineUpscale)
        inimage = inimage.convert("RGB")
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.pipelineUpscale)
        with torch.autocast(self.inferencedevice):
            image = self.pipelineUpscale.pipeline(image=inimage, prompt=prompt).images[0]
        return image