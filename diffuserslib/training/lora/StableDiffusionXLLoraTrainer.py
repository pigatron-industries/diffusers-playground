from .LoraTrainingParameters import LoraTrainingParameters
from ..DiffusersTrainer import DiffusersTrainer

import os
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr

from peft import LoraConfig, set_peft_model_state_dict


logger = get_logger(__name__)


class StableDiffusionLoraTrainer(DiffusersTrainer):

    def __init__(self, params:LoraTrainingParameters):
        self.params = params
        super().__init__(params)


    def load_models(self):
        super().load_models()
        self.text_encoder_trainers.append(self.createTextEncoderTrainer("tokenizer_2", "text_encoder_2"))


    def train(self):
        if self.params.seed is not None:
            set_seed(self.params.seed)

        #  Generate class images if prior preservation is enabled.
        if self.params.priorPreservation:
            self.generate_class_images()

        self.load_models()
        # self.load_validation_pipeline()
        # self.init_tokenizer()
        self.save_params()

        # Freeze vae and unet and text_encoders
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.text_encoder.requires_grad_(False)
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=torch.float32)  # The VAE is always in float32 to avoid NaN losses.
        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        if self.params.enableXformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if self.params.gradientCheckpointing:
            self.unet.enable_gradient_checkpointing()
            if(self.params.trainTextEncoder):
                for text_encoder_trainer in self.text_encoder_trainers:
                    text_encoder_trainer.text_encoder.gradient_checkpointing_enable()

        # add new LoRA weights to the attention layers
        unet_lora_config = LoraConfig(
            r=self.params.rank,
            lora_alpha=self.params.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config)

        # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
        # So, instead, we monkey-patch the forward calls of its attention-blocks.
        if self.params.trainTextEncoder:
            text_lora_config = LoraConfig(
                r=self.params.rank,
                lora_alpha=self.params.rank,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder_trainer.text_encoder.add_adapter(text_lora_config)

        self.scaleLearningRate()

        # Make sure the trainable params are in float32.
        if self.params.mixedPrecision == "fp16":
            models = [self.unet]
            if self.params.trainTextEncoder:
                models.extend([text_encoder_trainer.text_encoder for text_encoder_trainer in self.text_encoder_trainers])
            cast_training_params(models, dtype=torch.float32)

        self.unet_lora_parameters = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        if self.params.trainTextEncoder:
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder_trainer.fetch_text_encoder_parameters()

        # Optimization parameters
        unet_lora_parameters_with_lr = {"params": self.unet_lora_parameters, "lr": self.params.learningRate}
        if self.params.trainTextEncoder:
            # different learning rate for text encoder and unet
            text_lora_parameters_one_with_lr = {
                "params": text_lora_parameters_one,
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }
            text_lora_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
                "weight_decay": args.adam_weight_decay_text_encoder,
                "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
            }
            params_to_optimize = [
                unet_lora_parameters_with_lr,
                text_lora_parameters_one_with_lr,
                text_lora_parameters_two_with_lr,
            ]
        else:
            params_to_optimize = [unet_lora_parameters_with_lr]


    def generate_class_images(self):
        # TODO
        pass


    def save_params(self):
        if self.params.outputDir is not None:
            os.makedirs(self.params.outputDir, exist_ok=True)
        trainparamsfile = f"{self.params.outputDir}/{self.params.outputPrefix}-params.json"
        with open(trainparamsfile, 'w') as f:
            f.write(self.params.toJson())