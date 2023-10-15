import logging
import math
import os
from typing import List

from .TrainingParameters import TrainingParameters
from .TextualInversionDataset import TextualInversionDataset
from .TextEncoderTrainer import TextEncoderTrainer

import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from dataclasses import dataclass
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from IPython.display import display

logger = get_logger(__name__)


class StableDiffusionEmbeddingTrainer():

    def __init__(self, params:TrainingParameters):
        if params.numVectors < 1:
            raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {self.params.numVectors}")
        self.params = params
        self.accelerator = Accelerator(
            gradient_accumulation_steps = params.gradientAccumulationSteps,
            project_config = ProjectConfiguration(project_dir=params.outputDir),
        )

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16


    def train(self):
        if self.params.seed is not None:
            set_seed(self.params.seed)

        if self.params.outputDir is not None:
            os.makedirs(self.params.outputDir, exist_ok=True)

        self.load_models()
        self.init_tokenizer()

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder_trainer.text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder_trainer.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.params.gradientCheckpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.unet.enable_gradient_checkpointing()
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder_trainer.text_encoder.gradient_checkpointing_enable()

        if self.params.enableXformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if self.params.scaleLearningRate:
            self.params.learningRate = (
                self.params.learningRate * self.params.gradientAccumulationSteps * self.params.batchSize * self.accelerator.num_processes
            )

        # Initialize the optimizer
        trainable_params = []
        for text_encoder_trainer in self.text_encoder_trainers:
            trainable_params += text_encoder_trainer.text_encoder.get_input_embeddings().parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,  # only optimize the embeddings
            lr=self.params.learningRate,
            betas=(self.params.adamBeta1, self.params.adamBeta2),
            weight_decay=self.params.adamWeightDecay,
            eps=self.params.adamEpsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = TextualInversionDataset(
            data_root=self.params.trainDataDir,
            size=self.params.resolution,
            placeholder_token=self.placeholder_token_string,
            repeats=self.params.repeats,
            learnable_property=self.params.learnableProperty,
            center_crop=self.params.centreCrop,
            set="train",
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params.batchSize, shuffle=True, num_workers=0
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.params.gradientAccumulationSteps)
        if self.params.maxSteps is None:
            self.params.maxSteps = self.params.numEpochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.params.learningRateSchedule,
            optimizer=self.optimizer,
            num_warmup_steps=self.params.learningRateWarmupSteps * self.accelerator.num_processes,
            num_training_steps=self.params.maxSteps * self.accelerator.num_processes,
            num_cycles=self.params.learningRateNumCycles,
        )

        # Prepare everything with our `accelerator`.
        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.text_encoder = self.accelerator.prepare(text_encoder_trainer.text_encoder)
        self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.train_dataloader, self.lr_scheduler)

        # Move vae and unet to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.params.gradientAccumulationSteps)
        if overrode_max_train_steps:
            self.params.maxSteps = self.params.numEpochs * num_update_steps_per_epoch
        self.params.numEpochs = math.ceil(self.params.maxSteps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("textual_inversion", config=vars(self.params))

        # Train!
        total_batch_size = self.params.batchSize * self.accelerator.num_processes * self.params.gradientAccumulationSteps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self.params.numEpochs}")
        logger.info(f"  Instantaneous batch size per device = {self.params.batchSize}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.params.gradientAccumulationSteps}")
        logger.info(f"  Total optimization steps = {self.params.maxSteps}")
        self.global_step = 0
        self.start_epoch = 0
    
        initial_global_step = 0
        self.progress_bar = tqdm(
            range(0, self.params.maxSteps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        # keep original embeddings as reference
        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.store_original_embeddings()

        # Initial validation to see what prompt looks liike without training
        if self.accelerator.is_main_process:
            if self.params.validationPrompt is not None and self.global_step % self.params.validationSteps == 0:
                self.log_validation(0)

        self.train_loop()


    def load_models(self):
        self.pipeline = DiffusionPipeline.from_pretrained(self.params.model, safety_checker=None, torch_dtype=self.weight_dtype)
        self.text_encoder_trainers = [self.createTextEncoderTrainer("tokenizer", "text_encoder")]
        self.noise_scheduler = self.pipeline.components["scheduler"]
        self.vae = self.pipeline.components["vae"]
        self.unet = self.pipeline.components["unet"]


    def createTextEncoderTrainer(self, tokenizer_subfolder:str, text_encoder_subfolder:str):
        return TextEncoderTrainer(
            self.pipeline.components[tokenizer_subfolder],
            self.pipeline.components[text_encoder_subfolder],
            self.accelerator)


    def train_loop(self):
        for epoch in range(self.start_epoch, self.params.numEpochs):
            # TODO train all text encoders
            self.text_encoder_trainers[0].text_encoder.train()          
            for step, batchitem in enumerate(self.train_dataloader):
                loss = self.train_step(batchitem)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.progress_bar.update(1)
                    self.global_step += 1
                    if self.global_step % self.params.saveSteps == 0:
                        self.save_progress()
                    if self.accelerator.is_main_process:
                        if self.params.validationPrompt is not None and self.global_step % self.params.validationSteps == 0:
                            self.log_validation(epoch)

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)
                if self.global_step >= self.params.maxSteps:
                    break

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:      
            self.save_progress()
        self.accelerator.end_training()


    def train_step(self, batchitem):
        with self.accelerator.accumulate(self.text_encoder_trainers[0].text_encoder):
            # Convert images to latent space
            latents = self.vae.encode(batchitem["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample().detach()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            text_encoder_conds = self.get_text_conds(batchitem)

            # Predict the noise residual
            model_pred = self.call_unet(noisy_latents, timesteps, text_encoder_conds, batchitem)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Let's make sure we don't update any embedding weights besides the newly added token
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder_trainer.restore_original_embeddings()

            return loss
        

    def get_text_conds(self, batch) -> List[BaseModelOutputWithPooling]:
        text_conds = []
        for text_encoder_trainer in self.text_encoder_trainers:
            tokenizer = text_encoder_trainer.tokenizer
            input_ids = tokenizer(batch["caption"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
            input_ids = input_ids.to(self.accelerator.device)
            prompt_embeds = text_encoder_trainer.text_encoder(input_ids, output_hidden_states=True)
            text_conds.append(prompt_embeds)
        return text_conds
    

    def call_unet(self, noisy_latents, timesteps, text_encoder_conds: List[BaseModelOutputWithPooling], batch):
        noisy_latents = noisy_latents.to(self.weight_dtype)
        return self.unet(noisy_latents, timesteps, text_encoder_conds[0][0]).sample


    def init_tokenizer(self):
        # Add the placeholder tokens in tokenizer
        placeholder_tokens = [self.params.placeholderToken]
        for i in range(1, self.params.numVectors):
            placeholder_tokens.append(f"{self.params.placeholderToken}_{i}")
        self.placeholder_token_string = " ".join(placeholder_tokens)

        for text_encoder_trainer in self.text_encoder_trainers:
            text_encoder_trainer.add_tokens(placeholder_tokens, self.params.initializerToken)


    def load_validation_pipeline(self):
        if self.params.validationModel is None:
            validationModel = self.params.model
        else:
            validationModel = self.params.validationModel
        pipeline = DiffusionPipeline.from_pretrained(
            validationModel,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder),
            tokenizer=self.text_encoder_trainers[0].tokenizer,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        return pipeline


    def log_validation(self, epoch):
        logger.info(
            f"Running validation for step {self.global_step}... \n Generating {self.params.numValidationImages} images with prompt:"
            f" {self.params.validationPrompt}."
        )
        pipeline = self.load_validation_pipeline()

        # run inference
        generator = None if self.params.validationSeed is None else torch.Generator(device=self.accelerator.device).manual_seed(self.params.validationSeed)
        images = []
        for _ in range(self.params.numValidationImages):
            image = pipeline(self.params.validationPrompt, 
                             negative_prompt = self.params.validationNegativePrompt,
                             num_inference_steps = self.params.validationInferenceSteps, 
                             guidance_scale=9.0,
                             generator = generator,
                             width = self.params.validationSize[0], 
                             height = self.params.validationSize[1]).images[0]
            display(image)
            images.append(image)

        del pipeline
        return images


    def save_progress(self):
        logger.info("Saving embeddings")

        filename = f"{self.params.outputPrefix}-{self.params.placeholderToken}-{self.params.numVectors}-{self.global_step}"
        weight_name = (
            f"{filename}.safetensors"
            if self.params.safetensors
            else f"{filename}.bin"
        )
        save_path = os.path.join(self.params.outputDir, weight_name)

        learned_embeds = (
            self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder)
            .get_input_embeddings()
            .weight[min(self.text_encoder_trainers[0].placeholder_token_ids) : max(self.text_encoder_trainers[0].placeholder_token_ids) + 1]
        )
        learned_embeds_dict = {self.params.placeholderToken: learned_embeds.detach().cpu()}

        if self.params.safetensors:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
