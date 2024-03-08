from .LoraTrainingParameters import LoraTrainingParameters
from ..DiffusersTrainer import DiffusersTrainer

import os
import torch
import torch.utils.data
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

        # Optimization parameters
        unet_lora_parameters_with_lr = {"params": self.unet_lora_parameters, "lr": self.params.learningRate}
        params_to_optimize = [unet_lora_parameters_with_lr]
        if self.params.trainTextEncoder:
            for text_encoder_trainer in self.text_encoder_trainers:
                params_to_optimize.append({
                    "params": text_encoder_trainer.fetch_text_encoder_parameters(),
                    "lr": self.params.learningRate,
                    "weight_decay": self.params.textEncoderWeightDecay,
                })

        # Optimizer creation
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(self.params.adamBeta1, self.params.adamBeta2),
            weight_decay =  self.params.adamWeightDecay,
            eps = self.params.adamEpsilon,
        )

        # Dataset and DataLoaders creation:
        self.train_dataset = DreamBoothDataset(
            instance_data_root=self.params.trainDataDir,
            instance_prompt=self.params.instancePrompt,
            class_prompt=self.params.classPrompt,
            class_data_root=self.params.classDir if self.params.priorPreservation else None,
            class_num=self.params.numClassImages,
            size=self.params.resolution,
            repeats=self.params.repeats,
            center_crop=self.params.centreCrop,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.params.batchSize,
            shuffle = True,
            collate_fn = lambda examples: self.collate(examples, self.params.priorPreservation),
            num_workers = 1,
        )

        # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
        # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
        # the redundant encoding.
        if not self.params.trainTextEncoder and not train_dataset.custom_instance_prompts:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
                args.instance_prompt, text_encoders, tokenizers
            )



    def collate(self, examples, with_prior_preservation=False):
        pixel_values = [example["instance_images"] for example in examples]
        prompts = [example["instance_prompt"] for example in examples]
        original_sizes = [example["original_size"] for example in examples]
        crop_top_lefts = [example["crop_top_left"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if with_prior_preservation:
            pixel_values += [example["class_images"] for example in examples]
            prompts += [example["class_prompt"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        batch = {
            "pixel_values": pixel_values,
            "prompts": prompts,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }
        return batch


    def generate_class_images(self):
        # TODO
        pass


    def save_params(self):
        if self.params.outputDir is not None:
            os.makedirs(self.params.outputDir, exist_ok=True)
        trainparamsfile = f"{self.params.outputDir}/{self.params.outputPrefix}-params.json"
        with open(trainparamsfile, 'w') as f:
            f.write(self.params.toJson())