from .StableDiffusionEmbeddingTrainer import StableDiffusionEmbeddingTrainer
from .EmbeddingTrainingParameters import EmbeddingTrainingParameters

from diffusers import DiffusionPipeline, DDPMScheduler


class NoWatermark:
    def apply_watermark(self, img):
        return img


class StableDiffusionXLEmbeddingTrainer(StableDiffusionEmbeddingTrainer):

    def __init__(self, params:EmbeddingTrainingParameters):
        super().__init__(params)


    def load_models(self):
        super().load_models()
        self.text_encoder_trainers.append(self.createTextEncoderTrainer("tokenizer_2", "text_encoder_2"))


    def load_validation_pipeline(self):
        if(self.params.numValidationImages > 0):
            super().load_validation_pipeline()
            self.validationPipeline.watermark = NoWatermark()


    def call_unet(self, noisy_latents, timesteps, text_encoder_conds, batch):
        noisy_latents = noisy_latents.to(self.weight_dtype)
        (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = self.pipeline.encode_prompt(prompt=batch["caption"])
        time_ids = self.pipeline._get_add_time_ids(list(batch["pixel_values"].shape[2:4]), [0,0], [1024,1024], dtype=self.weight_dtype,
                                                text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim).to(self.accelerator.device)

        cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
        # prompt_embeds = torch.cat(text_encoder_conds, dim=-1).to(self.weight_dtype)
        return self.unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=cond_kwargs).sample


    def update_validation_pipeline(self):
        self.validationPipeline.text_encoder = self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder)
        self.validationPipeline.text_encoder_2 = self.accelerator.unwrap_model(self.text_encoder_trainers[1].text_encoder)
        self.validationPipeline.tokenizer = self.text_encoder_trainers[0].tokenizer
        self.validationPipeline.tokenizer_2 = self.text_encoder_trainers[1].tokenizer


    def to_state_dict(self, learned_embeds):
        """ convert embed to state dict for saving to file """
        return {
            "clip_l": learned_embeds[0],
            "clip_g": learned_embeds[1]
        }