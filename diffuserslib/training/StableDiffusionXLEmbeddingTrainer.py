from .StableDiffusionEmbeddingTrainer import StableDiffusionEmbeddingTrainer, TextEncoderTrainer
from .TrainingParameters import TrainingParameters
from .TextEncoderTrainer import TextEncoderTrainer

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DDPMScheduler
)


class StableDiffusionXLEmbeddingTrainer(StableDiffusionEmbeddingTrainer):

    def __init__(self, params:TrainingParameters):
        super().__init__(params)


    def load_models(self):
        self.pipeline = DiffusionPipeline.from_pretrained(self.params.model, safety_checker=None, torch_dtype=self.weight_dtype)
        self.text_encoder_trainers = [self.createTextEncoderTrainer("tokenizer", "text_encoder"), self.createTextEncoderTrainer("tokenizer_2", "text_encoder_2")]
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.params.model, subfolder="scheduler")
        self.vae = self.pipeline.components["vae"]
        self.unet = self.pipeline.components["unet"]


    def call_unet(self, noisy_latents, timesteps, text_encoder_conds, batch):
        noisy_latents = noisy_latents.to(self.weight_dtype)
        (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = self.pipeline.encode_prompt(prompt=batch["caption"])
        time_ids = self.pipeline._get_add_time_ids(list(batch["pixel_values"].shape[2:4]), [0,0], [1024,1024], dtype=self.weight_dtype).to(self.accelerator.device)

        cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
        # prompt_embeds = torch.cat(text_encoder_conds, dim=-1).to(self.weight_dtype)
        return self.unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=cond_kwargs).sample


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