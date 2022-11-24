import torch
import random
import os
from huggingface_hub import login
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor, CLIPModel

DEFAULT_AUTOENCODER_MODEL = 'stabilityai/sd-vae-ft-mse'
DEFAULT_TEXTTOIMAGE_MODEL = 'runwayml/stable-diffusion-v1-5'
DEFAULT_INPAINT_MODEL = 'runwayml/stable-diffusion-inpainting'
DEFAULT_CLIP_MODEL = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
MAX_SEED = 4294967295


# Use to bypass safety checker
def dummy(images, **kwargs):
    return images, False


class DiffusersLib:

    def __init__(self, device = "cuda"):
        self.device = "cuda"
        self.textToImagePipeline = None
        self.imageToImagePipeline = None
        self.inpaintingPipeline = None
        self.vae = None
        self.tokenizer = None
        self.text_encoder = None


    def loginHuggingFace(self, token):
        login(token=token)


    def loadAutoencoder(self, model = DEFAULT_AUTOENCODER_MODEL):
        self.vae = AutoencoderKL.from_pretrained(model)


    def loadTextEmbedding(self, embed_file):
        learned_embeds = torch.load(embed_file, map_location="cpu")
        trained_token = list(learned_embeds.keys())[0]
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
        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder')
        for embed_file in os.listdir(directory):
            file_path = directory + '/' + embed_file
            print(file_path)
            self.loadTextEmbedding(file_path)


    def loadCLIP(self, model=DEFAULT_CLIP_MODEL):
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(DEFAULT_CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL, torch_dtype=torch.float16)


    def createGenerator(self, seed=None):
        if(seed is None):
            seed = random.randInt(0, MAX_SEED)
        return torch.Generator(device=self.device).manual_seed(seed)


    def createTextToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL, custom_pipeline=None):
        print(f"Creating text to image pipeline from model {model}")
        args = {}
        args['safety_checker'] = dummy
        args['torch_dtype'] = torch.float16
        if(self.vae is not None):
            args['vae'] = self.vae
        if(self.tokenizer is not None):
            args['tokenizer'] = self.tokenizer
        if(self.text_encoder is not None):
            args['text_encoder'] = self.text_encoder
        if(custom_pipeline is not None and custom_pipeline != ''):
            args['custom_pipeline'] = custom_pipeline
        if(custom_pipeline == 'clip_guided_stable_diffusion'):
            args['feature_extractor'] = self.feature_extractor
            args['clip_model'] = self.clip_model

        self.textToImagePipeline = DiffusionPipeline.from_pretrained(model, **args)
        self.textToImagePipeline.enable_attention_slicing()
        self.textToImagePipeline = self.textToImagePipeline.to(self.device)


    def textToImage(self, prompt, negprompt, steps, scale, width, height, seed=None):
        generator = self.createGenerator(seed)
        with torch.autocast(self.device):
            image = self.textToImagePipeline(prompt, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        return image, seed


    def createImageToImagePipeline(self, model=DEFAULT_TEXTTOIMAGE_MODEL):
        print(f"Creating image to image pipeline from model {model}")
        args = {}
        args['safety_checker'] = dummy
        args['torch_dtype'] = torch.float16
        if(self.vae is not None):
            args['vae'] = self.vae
        if(self.tokenizer is not None):
            args['tokenizer'] = self.tokenizer
        if(self.text_encoder is not None):
            args['text_encoder'] = self.text_encoder

        self.imageToImagePipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model, **args)
        self.imageToImagePipeline.enable_attention_slicing()
        self.imageToImagePipeline = self.imageToImagePipeline.to(self.device)


    def ImageToImage(self, inimage, prompt, negprompt, steps, scale, width, height, seed=None):
        inimage = inimage.convert("RGB")
        generator = self.createGenerator(seed)
        with torch.autocast(self.device):
            image = self.imageToImagePipeline(prompt, init_image=inimage, negative_prompt=negprompt, num_inference_steps=steps, guidance_scale=scale, width=width, height=height, generator=generator).images[0]
        return image, seed


    def createInpaintPipeline(self, model=DEFAULT_INPAINT_MODEL):
        print(f"Creating inpainting pipeline from model {model}")


