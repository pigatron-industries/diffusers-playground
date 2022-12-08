import torch
import random
import os
import sys
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPModel
from .DiffusersModelPresets import DiffusersModelList

DEFAULT_AUTOENCODER_MODEL = 'stabilityai/sd-vae-ft-mse'
DEFAULT_TEXTTOIMAGE_MODEL = 'runwayml/stable-diffusion-v1-5'
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

    def __init__(self, device = DEFAULT_DEVICE):
        self.device = device
        self.textToImagePipeline = None
        self.imageToImagePipeline = None
        self.inpaintingPipeline = None
        self.upscalePipeline = None
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None
        self.schedulerClass = None
        self.presets = DiffusersModelList()


    def addPresets(self, presets):
        self.presets.addModels(presets)


    def loadAutoencoder(self, model = DEFAULT_AUTOENCODER_MODEL):
        self.vae = AutoencoderKL.from_pretrained(model)


    def loadTextEmbedding(self, embed_file):
        learned_embeds = torch.load(embed_file, map_location="cpu")
        trained_token = list(learned_embeds.keys())[0]
        print(f"loaded embedding token {trained_token}")
        learned_embed = learned_embeds[trained_token]
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        learned_embed.to(dtype)
        num_added_tokens = self.tokenizer.add_tokens(trained_token) # can replace token with something else if needed
        if(num_added_tokens == 0):
            raise ValueError(f"The tokenizer already contains the token {trained_token}")
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_id = self.tokenizer.convert_tokens_to_ids(trained_token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = learned_embed


    def loadTextEmbeddings(self, directory, model=DEFAULT_TEXTTOIMAGE_MODEL):
        print('loading text embeddings')
        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder')
        for embed_file in os.listdir(directory):
            file_path = directory + '/' + embed_file
            self.loadTextEmbedding(file_path)


    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(DEFAULT_CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL, torch_dtype=torch.float16)


    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randInt(0, MAX_SEED)
        return torch.Generator(device=self.device).manual_seed(seed), seed


    def loadScheduler(self, schedulerClass, pipeline):
        if (isinstance(schedulerClass, str)):
            self.schedulerClass = str_to_class(schedulerClass)
        else:
            self.schedulerClass = schedulerClass
        pipeline.scheduler = schedulerClass.from_config(pipeline.scheduler.config)


    def createTextToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL, custom_pipeline=None):
        print(f"Creating text to image pipeline from model {model}")
        preset = self.presets.getModel(model)
        args = {}
        args['safety_checker'] = None
        args['torch_dtype'] = torch.float16
        if(preset.fp16):
            args['revision'] = 'fp16'
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        if(self.tokenizer is not None):
            args['tokenizer'] = self.tokenizer
        if(self.text_encoder is not None):
            args['text_encoder'] = self.text_encoder
        if(custom_pipeline is not None and custom_pipeline != ''):
            args['custom_pipeline'] = custom_pipeline
        if(custom_pipeline == 'clip_guided_stable_diffusion'):
            args['feature_extractor'] = self.feature_extractor
            args['clip_model'] = self.clip_model
        self.textToImagePipeline = DiffusionPipeline.from_pretrained(model, **args).to(self.device)
        self.textToImagePipeline.enable_attention_slicing()


    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None, scheduler=None):
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.textToImagePipeline)
        with torch.autocast(self.device):
            image = self.textToImagePipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        return image, seed


    def createImageToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL):
        print(f"Creating image to image pipeline from model {model}")
        preset = self.presets.getModel(model)
        args = {}
        args['safety_checker'] = None
        args['torch_dtype'] = torch.float16
        if(preset.fp16):
            args['revision'] = 'fp16'
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        if(self.tokenizer is not None):
            args['tokenizer'] = self.tokenizer
        if(self.text_encoder is not None):
            args['text_encoder'] = self.text_encoder
        self.imageToImagePipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model, **args).to(self.device)
        self.imageToImagePipeline.enable_attention_slicing()


    def imageToImage(self, inimage, prompt, negprompt, strength, scale, seed=None, scheduler=None):
        inimage = inimage.convert("RGB")
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.imageToImagePipeline)
        with torch.autocast(self.device):
            image = self.imageToImagePipeline(prompt, init_image=inimage, negative_prompt=negprompt, strength=strength, guidance_scale=scale, generator=generator).images[0]
        return image, seed


    def createInpaintPipeline(self, model=DEFAULT_INPAINT_MODEL, fp16revision=True):
        print(f"Creating inpainting pipeline from model {model}")
        preset = self.presets.getModel(model)
        args = {}
        args['safety_checker'] = None
        args['torch_dtype'] = torch.float16
        if(preset.fp16):
            args['revision'] = 'fp16'
        if(preset.vae is not None):
            args['vae'] = AutoencoderKL.from_pretrained(preset.vae)
        if(self.tokenizer is not None):
            args['tokenizer'] = self.tokenizer
        if(self.text_encoder is not None):
            args['text_encoder'] = self.text_encoder
        self.inpaintingPipeline = StableDiffusionInpaintPipeline.from_pretrained(model, **args).to(self.device)
        self.inpaintingPipeline.enable_attention_slicing()


    def inpaint(self, inimage, maskimage, prompt, negprompt, steps, scale, seed=None, scheduler=None):
        inimage = inimage.convert("RGB")
        maskimage = maskimage.convert("RGB")
        generator, seed = self.createGenerator(seed)
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.inpaintingPipeline)
        with torch.autocast(self.device):
            image = self.inpaintingPipeline(prompt, image=inimage, mask_image=maskimage, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]
        return image, seed


    def createUpscalePipeline(self, model=DEFAULT_UPSCALE_MODEL, fp16revision=True):
        print(f"Creating upscale pipeline from model {model}")
        presets = self.presets.getModel(model)
        args = {}
        args['torch_dtype'] = torch.float16
        if(presets.fp16):
            args['revision'] = 'fp16'
        self.upscalePipeline = StableDiffusionUpscalePipeline.from_pretrained(model, **args).to(self.device)
        self.upscalePipeline.enable_attention_slicing()


    def upscale(self, inimage, prompt, scheduler=None):
        inimage = inimage.convert("RGB")
        if(scheduler is not None):
            self.loadScheduler(scheduler, self.upscalePipeline)
        with torch.autocast(self.device):
            image = self.upscalePipeline(image=inimage, prompt=prompt).images[0]
        return image