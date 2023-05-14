import torch
import os
import sys
import gc
from typing import Dict
from diffusers import ( DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, 
                        StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, 
                        StableDiffusionImageVariationPipeline, StableDiffusionInstructPix2PixPipeline,
                        ControlNetModel, StableDiffusionControlNetPipeline)
from diffusers.models import AutoencoderKL
from transformers import CLIPFeatureExtractor, CLIPModel
from .TextEmbedding import TextEmbeddings
from .LORA import LORA
from .arch import *
from .BaseModelData import BaseModelData
from ..models.DiffusersModelPresets import DiffusersModelList
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


class DiffusersPipelines:
    def __init__(self, localmodelpath = '', device = DEFAULT_DEVICE, safety_checker = True, common_modifierdict = None, custom_pipeline = None, cache_dir = None):
        self.localmodelpath: str = localmodelpath
        self.localmodelcache: str = getModelsDir()
        self.device: str = device
        self.inferencedevice: str = 'cpu' if self.device == 'mps' else self.device
        self.safety_checker: bool = safety_checker
        if (common_modifierdict is None):
            self.common_modifierdict = {}
        else:
            self.common_modifierdict = common_modifierdict
        self.custom_pipeline = custom_pipeline
        self.cache_dir = cache_dir
        self.lora_use = []

        self.pipeline: DiffusersPipelineWrapper = None

        self.vae = None
        self.baseModelData: Dict[str, BaseModelData] = {}
        self.presets = DiffusersModelList()


    def loadPresetFile(self, filepath):
        self.presets.load_from_file(filepath)


    def addPresets(self, presets):
        self.presets.addModels(presets)


    def addPreset(self, modelid, base, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        self.presets.addModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)


    def getModifierDict(self, base):
        modifiers = self.common_modifierdict.copy()
        modifiers.update(self.baseModelData[base].modifierdict)
        return modifiers


    def loadAutoencoder(self, model = DEFAULT_AUTOENCODER_MODEL):
        self.vae = AutoencoderKL.from_pretrained(model)


    def getBaseModelData(self, base):
        if base not in self.baseModelData:
            self.baseModelData[base] = BaseModelData(base, TextEmbeddings(base))
        return self.baseModelData[base]
            

    #=============== TEXT EMBEDDING ==============

    def loadTextEmbeddings(self, directory):
        for path, base in getPathsFiles(f"{directory}/*/"):
            baseModelData = self.getBaseModelData(base)
            baseModelData.textembeddings.load_directory(path, base)
            baseModelData.modifierdict = baseModelData.textembeddings.modifiers


    def loadTextEmbedding(self, path, base, token=None):
        base = self.getBaseModelData(base)
        base.textembeddings.load_file(path, token)


    def _addTextEmbeddingsToPipeline(self, pipeline: DiffusersPipelineWrapper):
        if (pipeline.preset.base in self.baseModelData):
            self.baseModelData[pipeline.preset.base].textembeddings.add_to_model(pipeline.pipeline.text_encoder, pipeline.pipeline.tokenizer)


    def processPrompt(self, prompt: str, pipeline: DiffusersPipelineWrapper):
        """ expands embedding tokens into multiple tokens, for each vector in embedding """
        if (pipeline.preset.base in self.baseModelData):
            prompt = self.baseModelData[pipeline.preset.base].textembeddings.process_prompt_and_add_tokens(prompt, pipeline)
        return prompt


    #=============== LORA ==============

    def loadLORAs(self, directory):
        for path, base in getPathsFiles(f"{directory}/*/"):
            baseModelData = self.getBaseModelData(base)
            for lora_path, lora_file in getPathsFiles(f"{path}/*"):
                if (lora_file.endswith('.bin') or lora_file.endswith('.safetensors')):
                    print(f"Adding available LORA file: {lora_file}")
                    lora = LORA.from_file(lora_file, lora_path)
                    baseModelData.loras[lora_file] = lora


    def loadLORA(self, lora_path, base):
        baseModelData = self.getBaseModelData(base)
        lora = LORA.from_file(lora_path, lora_path)
        baseModelData.loras[lora_path] = lora


    def useLORAs(self, lora_use = []):
        if (set(lora_use) != set(self.lora_use)):
            self.lora_use = lora_use
            self.pipeline = None


    def getLORAList(self, model):
        base = self.getModel(model).base
        return list(self.getBaseModelData(base).loras.keys())


    def _addLORAsToPipeline(self, pipeline: DiffusersPipelineWrapper):
        for lora_use in self.lora_use:
            self._addLORAToPipeline(pipeline, lora_use.name, lora_use.weight)


    def _addLORAToPipeline(self, pipeline: DiffusersPipelineWrapper, lora_name, weight=1):
        if (pipeline.preset.base in self.baseModelData and lora_name in self.baseModelData[pipeline.preset.base].loras):
            print(f"Loading LORA {lora_name}")
            self.baseModelData[pipeline.preset.base].loras[lora_name].add_to_model(pipeline.pipeline, weight=weight, device=self.device)

    #=============== MODEL MERGING ==============

    def mergeModel(self, modelid, weight=0.5):
        preset = self.getModel(modelid)
        mergePipeline = self.pipeline.__class__(preset=preset, device=self.device, safety_checker=self.safety_checker)
        for moduleName in self.pipeline.pipeline.config.keys():
            module1 = getattr(self.pipeline.pipeline, moduleName)
            module2 = getattr(mergePipeline.pipeline, moduleName)
            if hasattr(module1, "state_dict") and hasattr(module2, "state_dict"):
                print(f"Merging state_dict of {moduleName}")
                updateStateDictFunc = getattr(module1, "load_state_dict")
                theta_0 = getattr(module1, "state_dict")()
                theta_1 = getattr(module2, "state_dict")()
                for key in theta_0.keys():
                    if key in theta_1:
                        theta_0[key] = (1 - weight) * theta_0[key] + weight * theta_1[key]
                for key in theta_1.keys():
                    if key not in theta_0:
                        theta_0[key] = theta_1[key]
                updateStateDictFunc(theta_0)
        if(isinstance(self.pipeline.preset.modelid, list)):
            self.pipeline.preset.modelid.append(modelid)
        else:
            self.pipeline.preset.modelid = [self.pipeline.preset.modelid, modelid]

    #===============  ==============

    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model)
        self.clip_model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16)


    def getModel(self, modelid):
        return self.presets.getModel(modelid)


    def latentsToImage(self, pipeline, latents):
        latents = 1 / 0.18215 * latents
        image = pipeline.vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        return pipeline.numpy_to_pil(image)


    #=============== LOAD PIPELINES ==============

    def createPipeline(self, pipelinetype, model, model_weight=None, **kwargs):
        if(isinstance(model, list)):
            preset = self.getModel(model[0])
        else:
            preset = self.getModel(model)
        pipelineWrapperClass = str_to_class(preset.pipelinetypes[pipelinetype]+"Wrapper")
        if(self.pipeline is not None and self.pipeline.isEqual(pipelineWrapperClass, model, **kwargs)):
            return self.pipeline
        
        print(f"Creating {pipelinetype} pipeline from model {model}")
        if (self.pipeline is not None):
            del self.pipeline
        gc.collect()
        torch.cuda.empty_cache()
        pipelineWrapper = pipelineWrapperClass(preset=preset, device=self.device, safety_checker=self.safety_checker, **kwargs)
        self.pipeline = pipelineWrapper

        if(isinstance(model, list)):
            for modelid in model[1:]:
                self.mergeModel(modelid, model_weight)
        self._addLORAsToPipeline(pipelineWrapper)
        return self.pipeline


    #=============== INFERENCE ==============

    def inference(self, pipeline:DiffusersPipelineWrapper, prompt, seed, **kwargs):
        prompt = self.processPrompt(prompt, pipeline)
        return pipeline.inference(prompt=prompt, seed=seed, **kwargs)
    

    def run(self, pipelinetype, model, prompt, **kwargs):
        pipelineWrapper = self.createPipeline(pipelinetype, model, **kwargs)
        prompt = self.processPrompt(prompt, pipelineWrapper)
        return pipelineWrapper.inference(prompt=prompt, **kwargs)


    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None, scheduler=None, model=None, model_weight=None, tiling=False, **kwargs):
        return self.run(pipelinetype="txt2img", model=model, model_weight=model_weight, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, 
                 width=width, height=height, seed=seed, scheduler=scheduler, tiling=tiling)


    def imageToImage(self, initimage, prompt, negprompt, strength, scale, seed=None, scheduler=None, model=None, model_weight=None, tiling=False, **kwargs):
        return self.run(pipelinetype="img2img", model=model, model_weight=model_weight, prompt=prompt, initimage=initimage, negprompt=negprompt, strength=strength, scale=scale, 
                        seed=seed, scheduler=scheduler, tiling=tiling)


    def inpaint(self, initimage, maskimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, model=None, model_weight=None, tiling=False, **kwargs):
        return self.run(pipelinetype="inpaint", model=model, model_weight=model_weight, prompt=prompt, initimage=initimage, maskimage=maskimage, width=initimage.width, height=initimage.height,
                        negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler, tiling=tiling)
    

    def textToImageControlNet(self, controlimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, model=None, model_weight=None, controlmodel=None, tiling=False, **kwargs):
        return self.run(pipelinetype="txt2img_controlnet", model=model, model_weight=model_weight, controlmodel=controlmodel, prompt=prompt, controlimage=controlimage, negprompt=negprompt, steps=steps, scale=scale, 
                        seed=seed, scheduler=scheduler, tiling=tiling)
    

    def imageToImageControlNet(self, initimage, controlimage, prompt, negprompt, strength, scale, seed=None, scheduler=None, model=None, model_weight=None, controlmodel=None, tiling=False, **kwargs):
        return self.run(pipelinetype="img2img_controlnet", model=model, model_weight=model_weight, controlmodel=controlmodel, prompt=prompt, initimage=initimage, controlimage=controlimage, negprompt=negprompt, 
                              strength=strength, scale=scale, seed=seed, scheduler=scheduler, tiling=tiling)
    

    def inpaintControlNet(self, initimage, maskimage, controlimage, prompt, negprompt, steps, scale, seed=None, scheduler=None, model=None, model_weight=None, controlmodel=None, tiling=False, **kwargs):
        return self.run(pipelinetype="inpaint_controlnet", model=model, model_weight=model_weight, controlmodel=controlmodel, prompt=prompt, initimage=initimage, maskimage=maskimage, controlimage=controlimage, 
                              negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler, tiling=tiling)


    def upscale(self, initimage, prompt, negprompt, scale, steps=40, seed=None, scheduler=None, model=None):
        return self.run(pipelinetype="upscale", model=model, initimage=initimage, prompt=prompt, negprompt=negprompt, steps=steps, scale=scale, seed=seed, scheduler=scheduler)