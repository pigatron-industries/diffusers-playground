from .StableDiffusionEmbeddingTrainer import StableDiffusionEmbeddingTrainer, TextEncoderTrainer
from .TrainingParameters import TrainingParameters

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

