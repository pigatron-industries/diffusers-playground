import logging
import math
import os

from .TrainingParameters import TrainingParameters
from .TextualInversionDataset import TextualInversionDataset

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

class TextEncoderTrainer():

    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder




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
        self.text_encoder_trainers[0].text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder_trainers[0].text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder_trainers[0].text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.params.gradientCheckpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.text_encoder_trainers[0].text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        if self.params.enableXformers:
            self.unet.enable_xformers_memory_efficient_attention()

        if self.params.scaleLearningRate:
            self.params.learningRate = (
                self.params.learningRate * self.params.gradientAccumulationSteps * self.params.batchSize * self.accelerator.num_processes
            )

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            self.text_encoder_trainers[0].text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=self.params.learningRate,
            betas=(self.params.adamBeta1, self.params.adamBeta2),
            weight_decay=self.params.adamWeightDecay,
            eps=self.params.adamEpsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = TextualInversionDataset(
            data_root=self.params.trainDataDir,
            tokenizer=self.text_encoder_trainers[0].tokenizer,
            size=self.params.resolution,
            placeholder_token=(" ".join(self.text_encoder_trainers[0].tokenizer.convert_ids_to_tokens(self.placeholder_token_ids))),
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
        self.text_encoder_trainers[0].text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder_trainers[0].text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

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
        self.orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder).get_input_embeddings().weight.data.clone()

        # Initial validation to see what prompt looks liike without training
        if self.accelerator.is_main_process:
            if self.params.validationPrompt is not None and self.global_step % self.params.validationSteps == 0:
                self.log_validation(0)

        self.train_loop()


    def train_loop(self):
        for epoch in range(self.start_epoch, self.params.numEpochs):
            self.text_encoder_trainers[0].text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)

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


    def train_step(self, batch):
        with self.accelerator.accumulate(self.text_encoder_trainers[0].text_encoder):
            # Convert images to latent space
            latents = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample().detach()
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
            text_encoder_conds = self.get_text_conds(batch)

            # Predict the noise residual
            model_pred = self.call_unet(noisy_latents, timesteps, text_encoder_conds, batch)

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
            index_no_updates = torch.ones((len(self.text_encoder_trainers[0].tokenizer),), dtype=torch.bool)
            index_no_updates[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1] = False

            with torch.no_grad():
                self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder).get_input_embeddings().weight[
                    index_no_updates
                ] = self.orig_embeds_params[index_no_updates]

            return loss


    def load_models(self):
        self.text_encoder_trainers = [TextEncoderTrainer(
            CLIPTokenizer.from_pretrained(self.params.model, subfolder="tokenizer"), 
            CLIPTextModel.from_pretrained(self.params.model, subfolder="text_encoder")
        )]
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.params.model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.params.model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.params.model, subfolder="unet")


    def get_text_conds(self, batch):
        return self.text_encoder_trainers[0].text_encoder(batch["input_ids"])[0].to(dtype=self.weight_dtype)
    

    def call_unet(self, noisy_latents, timesteps, text_encoder_conds, batch):
        return self.unet(noisy_latents, timesteps, text_encoder_conds).sample


    def init_tokenizer(self):
        # Add the placeholder tokens in tokenizer
        placeholder_tokens = [self.params.placeholderToken]
        additional_tokens = []
        for i in range(1, self.params.numVectors):
            additional_tokens.append(f"{self.params.placeholderToken}_{i}")
        placeholder_tokens += additional_tokens
        num_added_tokens = self.text_encoder_trainers[0].tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.params.numVectors:
            raise ValueError(f"The tokenizer already contains the token {self.params.placeholderToken}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        self.placeholder_token_ids = self.text_encoder_trainers[0].tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # Convert the initializer_token to ids
        initializer_token_ids = self.text_encoder_trainers[0].tokenizer.encode(self.params.initializerToken, add_special_tokens=False)
        if len(initializer_token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")      # TODO use all tokens in the initializer instead of erroring
        initializer_token_id = initializer_token_ids[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        self.text_encoder_trainers[0].text_encoder.resize_token_embeddings(len(self.text_encoder_trainers[0].tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = self.text_encoder_trainers[0].text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for placeholder_token_id in self.placeholder_token_ids:
                token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()


    def log_validation(self, epoch):
        logger.info(
            f"Running validation for step {self.global_step}... \n Generating {self.params.numValidationImages} images with prompt:"
            f" {self.params.validationPrompt}."
        )
        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            self.params.model,
            text_encoder=self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder),
            tokenizer=self.text_encoder_trainers[0].tokenizer,
            unet=self.unet,
            vae=self.vae,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = None if self.params.validationSeed is None else torch.Generator(device=self.accelerator.device).manual_seed(self.params.validationSeed)
        images = []
        for _ in range(self.params.numValidationImages):
            image = pipeline(self.params.validationPrompt, 
                             negative_prompt = self.params.validationNegativePrompt,
                             num_inference_steps = self.params.validationSteps, 
                             guidance_scale=9.0,
                             generator = generator,
                             width = self.params.resolution, 
                             height = self.params.resolution).images[0]
            display(image)
            images.append(image)

        del pipeline
        return images


    def save_progress(self):
        logger.info("Saving embeddings")

        weight_name = (
            f"learned_embeds-steps-{self.global_step}.safetensors"
            if self.params.safetensors
            else f"learned_embeds-steps-{self.global_step}.bin"
        )
        save_path = os.path.join(self.params.outputDir, weight_name)

        learned_embeds = (
            self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder)
            .get_input_embeddings()
            .weight[min(self.placeholder_token_ids) : max(self.placeholder_token_ids) + 1]
        )
        learned_embeds_dict = {self.params.placeholderToken: learned_embeds.detach().cpu()}

        if self.params.safetensors:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
