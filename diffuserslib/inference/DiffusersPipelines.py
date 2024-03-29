import torch
import os
import sys
import gc
from typing import Dict, Tuple, List, Self
from PIL import Image
from diffusers.models import AutoencoderKL
from transformers import CLIPFeatureExtractor, CLIPModel
from .TextEmbedding import TextEmbeddings
from .LORA import LORA
from .arch import *
from .BaseModelData import BaseModelData
from ..models.DiffusersModelPresets import DiffusersModelList
from ..ModelUtils import getModelsDir, downloadModel, convertToDiffusers
from ..FileUtils import getPathsFiles
from numpy import ndarray

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
    pipelines:Self|None = None

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
        self.presets.addModel(modelid, base, modeltype = types[0], revision=revision, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath, pipelinetypes=pipelinetypes)


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


    def getEmbeddingTokens(self, base) -> List[str]:
        return list(self.getBaseModelData(base).textembeddings.embeddings.keys())


    #=============== LORA ==============

    def loadLORAs(self, directory):
        for path, base in getPathsFiles(f"{directory}/*/"):
            print(f"Loading LORAs for base {base} from {path}")
            baseModelData = self.getBaseModelData(base)
            for lora_path, lora_file in getPathsFiles(f"{path}/*") + getPathsFiles(f"{path}/**/*"):
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
    

    def getLORAsByBase(self, base):
        return list(self.getBaseModelData(base).loras.keys())


    def _getLORAs(self, params:GenerationParameters) -> Tuple[List[LORA], List[float]]:
        if(self.pipeline is None or self.pipeline.initparams.modelConfig is None):
            return [], []
        base = self.pipeline.initparams.modelConfig.base

        loras = []
        weights = []
        for loraparams in params.loras:
            if (base in self.baseModelData and loraparams.name in self.baseModelData[base].loras.loras):
                loras.append(self.baseModelData[base].loras[loraparams.name])
                weights.append(loraparams.weight)
        return loras, weights


    def processPrompt(self, params:GenerationParameters, pipeline:DiffusersPipelineWrapper):
        """ expands embedding tokens into multiple tokens, for each vector in embedding """
        if (pipeline.initparams.modelConfig is not None and pipeline.initparams.modelConfig.base in self.baseModelData):
            baseData = self.baseModelData[pipeline.initparams.modelConfig.base]
            prompt = baseData.textembeddings.process_prompt_and_add_tokens(params.original_prompt, pipeline)
            loras, weights = self._getLORAs(params)
            prompt = baseData.loras.process_prompt_and_add_loras(params.prompt, pipeline, loras, weights)
        
        return prompt


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


    def getModel(self, modelid:str) -> DiffusersModel:
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

        if (self.pipeline is not None):
            del self.pipeline

        pipelineWrapperClass = str_to_class(params.modelConfig.pipelinetypes[params.generationtype]+"Wrapper")
        pipelineWrapper = pipelineWrapperClass(device=self.device, params=params)
        self.pipeline = pipelineWrapper
        
        if(len(params.models) > 1):
            for modelparams in params.models[1:]:
                self.mergeModel(modelparams.name, modelparams.weight, params)
        return self.pipeline
    
    def loadModelConfigs(self, params:GenerationParameters):
        params.modelConfig = self.getModel(params.models[0].name)
        for controlimage in params.controlimages:
            if (controlimage.model is not None):
                controlimage.modelConfig = self.getModel(controlimage.model)
        return params


    #=============== INFERENCE ==============

    def generate(self, params:GenerationParameters) -> Tuple[Image.Image|ndarray, int]:
        params.safetychecker = self.safety_checker
        pipelineWrapper = self.createPipeline(params)
        params.prompt = self.processPrompt(params, pipelineWrapper)
        image, seed = pipelineWrapper.inference(params)
        gc.collect()
        torch.mps.empty_cache()
        torch.cuda.empty_cache()
        return image, seed


    def interrupt(self):
        if (self.pipeline is not None):
            self.pipeline.interrupt()
