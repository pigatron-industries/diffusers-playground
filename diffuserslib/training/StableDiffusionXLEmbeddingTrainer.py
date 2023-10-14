from .StableDiffusionEmbeddingTrainer import StableDiffusionEmbeddingTrainer, TextEncoderTrainer
from .TrainingParameters import TrainingParameters

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)


class StableDiffusionXLEmbeddingTrainer(StableDiffusionEmbeddingTrainer):

    def __init__(self, params:TrainingParameters):
        super().__init__(params)


    def load_models(self):
        self.text_encoder_trainers = [
            TextEncoderTrainer(
                CLIPTokenizer.from_pretrained(self.params.model, subfolder="tokenizer"), 
                CLIPTextModel.from_pretrained(self.params.model, subfolder="text_encoder")
            ),
            TextEncoderTrainer(
                CLIPTokenizer.from_pretrained(self.params.model, subfolder="tokenizer_2"),
                CLIPTextModel.from_pretrained(self.params.model, subfolder="text_encoder_2")
            )
        ]
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.params.model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.params.model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.params.model, subfolder="unet")

    def call_unet(self, noisy_latents, timesteps, text_encoder_conds, batch):
        noisy_latents = noisy_latents.to(self.weight_dtype)
        text_embedding = torch.cat(text_encoder_conds, dim=-1).to(self.weight_dtype)
        # added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return self.unet(noisy_latents, timesteps, text_embedding).sample
