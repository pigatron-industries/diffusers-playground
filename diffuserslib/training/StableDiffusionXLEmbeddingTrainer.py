from .StableDiffusionEmbeddingTrainer import StableDiffusionEmbeddingTrainer, TextEncoderTrainer
from .TrainingParameters import TrainingParameters
from .TextEncoderTrainer import TextEncoderTrainer

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    EulerDiscreteScheduler
)


class StableDiffusionXLEmbeddingTrainer(StableDiffusionEmbeddingTrainer):

    def __init__(self, params:TrainingParameters):
        super().__init__(params)


    def load_models(self):
        self.text_encoder_trainers = [self.createTextEncoderTrainer("tokenizer", "text_encoder"), 
                                      self.createTextEncoderTrainer("tokenizer_2", "text_encoder_2")]
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.params.model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.params.model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.params.model, subfolder="unet")


    def call_unet(self, noisy_latents, timesteps, text_encoder_conds, batch):
        noisy_latents = noisy_latents.to(self.weight_dtype)
        text_embedding = torch.cat(text_encoder_conds, dim=-1).to(self.weight_dtype)
        # added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return self.unet(noisy_latents, timesteps, text_embedding).sample


    def load_validation_pipeline(self):
        if self.params.validationModel is None:
            validationModel = self.params.model
        else:
            validationModel = self.params.validationModel
        pipeline = DiffusionPipeline.from_pretrained(
            validationModel,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder),
            text_encoder_2=self.accelerator.unwrap_model(self.text_encoder_trainers[1].text_encoder),
            tokenizer=self.text_encoder_trainers[0].tokenizer,
            tokenizer_2=self.text_encoder_trainers[1].tokenizer,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        return pipeline