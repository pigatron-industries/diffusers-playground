import torch
import os
import sys
import gc
from typing import Dict, Tuple, List
from PIL import Image
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

        self.pipeline: DiffusersPipelineWrapper|None = None

        self.vae = None
        self.baseModelData: Dict[str, BaseModelData] = {}
        self.presets = DiffusersModelList()


    def loadPresetFile(self, filepath):
        self.presets.load_from_file(filepath)


    def addPresets(self, presets):
        self.presets.addModels(presets)


    def addPreset(self, types, base, modelid, revision=None, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        pipelinetypes = {}
        for type in types:
            pipelinetypes[type] = self.presets.getBaseModel(base).pipelinetypes[type]
        self.presets.addModel(modelid, base, revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath, pipelinetypes=pipelinetypes)


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


    def processPrompt(self, prompt: str, pipeline: DiffusersPipelineWrapper):
        """ expands embedding tokens into multiple tokens, for each vector in embedding """
        if (pipeline.params.modelConfig.base in self.baseModelData):
            prompt = self.baseModelData[pipeline.params.modelConfig.base].textembeddings.process_prompt_and_add_tokens(prompt, pipeline)
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


    def getLORAList(self, model):
        base = self.getModel(model).base
        return list(self.getBaseModelData(base).loras.keys())


    def _addLORAsToPipeline(self, params:GenerationParameters):
        for loraparams in params.loras:
            self._addLORAToPipeline(loraparams.name, loraparams.weight)


    def _addLORAToPipeline(self, lora_name, weight:float=1.0):
        if (self.pipeline.params.modelConfig.base in self.baseModelData and lora_name in self.baseModelData[self.pipeline.params.modelConfig.base].loras):
            print(f"Loading LORA {lora_name}")
            self.baseModelData[self.pipeline.params.modelConfig.base].loras[lora_name].add_to_model(self.pipeline.pipeline, weight=weight, device=self.device)

    #=============== MODEL MERGING ==============

    def mergeModel(self, modelid:str, weight:float, params:GenerationParameters):
        print(f"Merging model {modelid}")
        preset = self.getModel(modelid)
        mergePipeline = self.pipeline.__class__(preset=preset, device=self.device, params=params)
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
        if(isinstance(self.pipeline.params.modelConfig.modelid, list)):
            self.pipeline.params.modelConfig.modelid.append(modelid)
        else:
            self.pipeline.params.modelConfig.modelid = [self.pipeline.params.modelConfig.modelid, modelid]

    #===============  ==============

    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model)
        self.clip_model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16)


    def getModel(self, modelid:str):
        return self.presets.getModel(modelid)


    def latentsToImage(self, pipeline, latents):
        latents = 1 / 0.18215 * latents
        image = pipeline.vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        return pipeline.numpy_to_pil(image)


    #=============== LOAD PIPELINES ==============

    def createPipeline(self, params:GenerationParameters):
        self.loadModelConfigs(params)
        if(self.pipeline is not None and self.pipeline.paramsMatch(params)):
            return self.pipeline

        print(f"Creating {params.getGenerationType()} pipeline from model {params.models[0].name}")
        if (self.pipeline is not None):
            del self.pipeline

        gc.collect()
        torch.cuda.empty_cache()

        # TODO load conditioning config/preset
        pipelineWrapperClass = str_to_class(params.modelConfig.pipelinetypes[params.getGenerationType()]+"Wrapper")
        pipelineWrapper = pipelineWrapperClass(device=self.device, params=params)
        self.pipeline = pipelineWrapper
        
        if(len(params.models) > 1):
            for modelparams in params.models[1:]:
                self.mergeModel(modelparams.name, modelparams.weight, params)
        self._addLORAsToPipeline(params)
        return self.pipeline
    
    def loadModelConfigs(self, params:GenerationParameters):
        params.modelConfig = self.getModel(params.models[0].name)
        for controlimage in params.controlimages:
            if (controlimage.model is not None):
                controlimage.modelConfig = self.getModel(controlimage.model)
        return params


    #=============== INFERENCE ==============

    def generate(self, params:GenerationParameters) -> Tuple[Image.Image, int]:
        params.safetychecker = self.safety_checker
        pipelineWrapper = self.createPipeline(params)
        params.prompt = self.processPrompt(params.original_prompt, pipelineWrapper)
        return pipelineWrapper.inference(params)
