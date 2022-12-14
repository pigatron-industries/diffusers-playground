import torch
import random
import os
import sys
import re
from diffusers import ( DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, 
                        StableDiffusionUpscalePipeline, StableDiffusionDepth2ImgPipeline, StableDiffusionImageVariationPipeline,
                        # Schedulers
                        DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, HeunDiscreteScheduler,
                        KDPM2DiscreteScheduler, KarrasVeScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,
                        KDPM2AncestralDiscreteScheduler, EulerAncestralDiscreteScheduler,
                        ScoreSdeVeScheduler, IPNDMScheduler, )
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPModel
from .DiffusersModelPresets import DiffusersModelList
from .StringUtils import findBetween
from .ModelUtils import getModelsDir, downloadModel, convertToDiffusers
from .ImageUtils import compositeImages

DEFAULT_AUTOENCODER_MODEL = 'stabilityai/sd-vae-ft-mse'
DEFAULT_TEXTTOIMAGE_MODEL = 'runwayml/stable-diffusion-v1-5'
DEFAULT_DEPTHTOIMAGE_MODEL = 'stabilityai/stable-diffusion-2-depth'
DEFAULT_INPAINT_MODEL = 'runwayml/stable-diffusion-inpainting'
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

    def __init__(self, localmodelpath = '', device = DEFAULT_DEVICE, safety_checker = True):
        self.localmodelpath = localmodelpath
        self.localmodelcache = getModelsDir()
        self.device = device
        self.inferencedevice = 'cpu' if self.device == 'mps' else self.device
        self.textToImagePipeline = None
        self.textToImagePreset = None
        self.imageToImagePipeline = None
        self.imageToImagePreset = None
        self.depthToImagePipeline = None
        self.depthToImagePreset = None
        self.inpaintingPipeline = None
        self.inpaintingPreset = None
        self.upscalePipeline = None
        self.upscalePreset = None
        self.vae = None
        self.tokenizers = {}
        self.text_encoders = {}
        self.safety_checker = safety_checker
        self.presets = DiffusersModelList()


    def addPresets(self, presets):
        self.presets.addModels(presets)

    
    def addPreset(self, modelid, base, fp16=True, stylephrase=None, vae=None, autocast=True, location='hf', modelpath=None):
        if(modelpath == None and location == 'local'):
            modelpath = self.localmodelpath + '/' + modelid
        self.presets.addModel(modelid, base, fp16=fp16, stylephrase=stylephrase, vae=vae, autocast=autocast, location=location, modelpath=modelpath)


    def loadAutoencoder(self, model = DEFAULT_AUTOENCODER_MODEL):
        self.vae = AutoencoderKL.from_pretrained(model)


    def loadTextEmbedding(self, embed_file, base, token=None):
        text_encoder = self.text_encoders[base]
        tokenizer = self.tokenizers[base]
        learned_embeds = torch.load(embed_file, map_location="cpu")
        
        if ('string_to_param' in learned_embeds):  # pt embedding
            string_to_token = learned_embeds['string_to_token']
            trained_token = list(string_to_token.keys())[0]
            string_to_param = learned_embeds['string_to_param']
            # I don't know why this tensor contains multiple dimensions, i'm just picking the first one for now
            learned_embed = string_to_param[trained_token][0] 
        else: # bin diffusers concept
            trained_token = list(learned_embeds.keys())[0]
            learned_embed = learned_embeds[trained_token]

        dtype = text_encoder.get_input_embeddings().weight.dtype
        learned_embed.to(dtype)
        if(token is None):
            token = trained_token
        print(f"loaded embedding token {token}")
        num_added_tokens = tokenizer.add_tokens(token)
        if(num_added_tokens == 0):
            raise ValueError(f"The tokenizer already contains the token {token}")
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = learned_embed


    def loadTextEmbeddings(self, directory, model=DEFAULT_TEXTTOIMAGE_MODEL):
        preset = self.getModel(model)
        embeddingspath = directory + '/' + preset.base
        print('loading text embeddings from path ' + embeddingspath)
        self.tokenizers[preset.base] = CLIPTokenizer.from_pretrained(preset.modelpath, subfolder='tokenizer', torch_dtype=torch.float16)
        self.text_encoders[preset.base] = CLIPTextModel.from_pretrained(preset.modelpath, subfolder='text_encoder', torch_dtype=torch.float16)
        for embed_file in os.listdir(embeddingspath):
            file_path = embeddingspath + '/' + embed_file
            print(f"loading embedding from file {embed_file}")
            token = findBetween(embed_file, '<', '>', True)
            if (file_path.endswith('.bin') or file_path.endswith('.pt')):
                self.loadTextEmbedding(file_path, preset.base, token)


    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(model)
        self.clip_model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16)


    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randint(0, MAX_SEED)
        return torch.Generator(device = self.inferencedevice).manual_seed(seed), seed


    def loadScheduler(self, schedulerClass, pipeline):
        if (isinstance(schedulerClass, str)):
            schedulerClass = str_to_class(schedulerClass)
        pipeline.scheduler = schedulerClass.from_config(pipeline.scheduler.config)


    def createArgs(self, preset):
        args = {}
        args['torch_dtype'] = torch.float16
        if (not self.safety_checker):
            args['safety_checker'] = None
        if(preset.fp16):
            args['revision'] = 'fp16'
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        if(preset.base in self.tokenizers):
            args['tokenizer'] = self.tokenizers[preset.base]
        if(preset.base in self.text_encoders):
            args['text_encoder'] = self.text_encoders[preset.base]
        return args


    def getModel(self, modelid):
        preset = self.presets.getModel(modelid)
        if(preset.location == 'url' and preset.modelpath.startswith('http')):
            localmodelpath = self.localmodelcache + '/' + modelid
            if (not os.path.isdir(localmodelpath)):
                downloadModel(preset.modelpath, modelid)
                convertToDiffusers(modelid)
            preset.modelpath = localmodelpath
        return preset


    def createTextToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL, custom_pipeline=None):
        if(self.textToImagePreset is not None and self.textToImagePreset.modelid == model):
            return
        print(f"Creating text to image pipeline from model {model}")
        self.textToImagePreset = self.getModel(model)
        args = self.createArgs(self.textToImagePreset)
        if(custom_pipeline is not None and custom_pipeline != ''):
            args['custom_pipeline'] = custom_pipeline
        if(custom_pipeline == 'clip_guided_stable_diffusion'):
            args['feature_extractor'] = self.feature_extractor
            args['clip_model'] = self.clip_model
        self.textToImagePipeline = DiffusionPipeline.from_pretrained(self.textToImagePreset.modelpath, **args).to(self.device)
        self.textToImagePipeline.enable_attention_slicing()


    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None, scheduler=None, model=None):
        if (self.textToImagePipeline is None):
            raise Exception('text to image pipeline not loaded')
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.textToImagePipeline)
        if(self.textToImagePreset.autocast):
            with torch.autocast(self.inferencedevice):
                image = self.textToImagePipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        else:
            image = self.textToImagePipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        return image, seed


    def createImageToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL):
        if(self.imageToImagePreset is not None and self.imageToImagePreset.modelid == model):
            return
        print(f"Creating image to image pipeline from model {model}")
        self.imageToImagePreset = self.getModel(model)
        args = self.createArgs(self.imageToImagePreset)
        self.imageToImagePipeline = StableDiffusionImg2ImgPipeline.from_pretrained(self.imageToImagePreset.modelpath, **args).to(self.device)
        self.imageToImagePipeline.enable_attention_slicing()


    def imageToImage(self, inimage, prompt, negprompt, strength, scale, seed=None, scheduler=None):
        if (self.imageToImagePipeline is None):
            raise Exception('image to image pipeline not loaded')
        inimage = inimage.convert("RGB")
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.imageToImagePipeline)
        with torch.autocast(self.inferencedevice):
            image = self.imageToImagePipeline(prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, generator=generator).images[0]
        return image, seed


    def createDepthToImagePipeline(self, model=DEFAULT_DEPTHTOIMAGE_MODEL):
        if(self.depthToImagePreset is not None and self.depthToImagePreset.modelid == model):
            return
        print(f"Creating depth to image pipeline from model {model}")
        self.depthToImagePreset = self.getModel(model)
        args = self.createArgs(self.depthToImagePreset)
        self.depthToImagePipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(self.depthToImagePreset.modelpath, **args).to(self.device)
        self.depthToImagePipeline.enable_attention_slicing()


    def depthToImage(self, inimage, prompt, negprompt, strength, scale, seed=None, scheduler=None):
        if (self.imageToImagePipeline is None):
            raise Exception('depth to image pipeline not loaded')
        inimage = inimage.convert("RGB")
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.depthToImagePipeline)
        with torch.autocast(self.inferencedevice):
            image = self.depthToImagePipeline(prompt, image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, generator=generator).images[0]
        return image, seed


    def createInpaintPipeline(self, model=DEFAULT_INPAINT_MODEL):
        if(self.inpaintingPreset is not None and self.inpaintingPreset.modelid == model):
            return
        print(f"Creating inpainting pipeline from model {model}")
        self.inpaintingPreset = self.getModel(model)
        args = self.createArgs(self.inpaintingPreset)
        self.inpaintingPipeline = StableDiffusionInpaintPipeline.from_pretrained(self.inpaintingPreset.modelpath, **args).to(self.device)
        self.inpaintingPipeline.enable_attention_slicing()


    def inpaint(self, initimage, maskimage, prompt, negprompt, steps, scale, seed=None, scheduler=None):
        if (self.inpaintingPipeline is None):
            raise Exception('inpainting pipeline not loaded')
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.inpaintingPipeline)
        with torch.autocast(self.inferencedevice):
            outimage = self.inpaintingPipeline(prompt, image=initimage.convert("RGB"), mask_image=maskimage.convert("RGB"), 
                                               negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return outimage, seed


    def createUpscalePipeline(self, model=DEFAULT_UPSCALE_MODEL):
        if(self.upscalePreset is not None and self.upscalePreset.modelid == model):
            return
        print(f"Creating upscale pipeline from model {model}")
        self.upscalePreset = self.getModel(model)
        args = {}
        args['torch_dtype'] = torch.float16
        if(self.upscalePreset.fp16):
            args['revision'] = 'fp16'
        self.upscalePipeline = StableDiffusionUpscalePipeline.from_pretrained(self.upscalePreset.modelpath, **args).to(self.device)
        self.upscalePipeline.enable_attention_slicing()


    def upscale(self, inimage, prompt, scheduler=None):
        if (self.upscalePipeline is None):
            raise Exception('upscale pipeline not loaded')
        inimage = inimage.convert("RGB")
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.upscalePipeline)
        with torch.autocast(self.inferencedevice):
            image = self.upscalePipeline(image=inimage, prompt=prompt).images[0]
        return image