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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
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

        # Load models
        tokenizer = CLIPTokenizer.from_pretrained(self.params.model, subfolder="tokenizer")
        noise_scheduler = DDPMScheduler.from_pretrained(self.params.model, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(self.params.model, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.params.model, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.params.model, subfolder="unet")

        if self.params.numVectors < 1:
            raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {self.params.numVectors}")

        # Add the placeholder tokens in tokenizer
        placeholder_tokens = [self.params.placeholderToken]
        additional_tokens = []
        for i in range(1, self.params.numVectors):
            additional_tokens.append(f"{self.params.placeholderToken}_{i}")
        placeholder_tokens += additional_tokens
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens != self.params.numVectors:
            raise ValueError(f"The tokenizer already contains the token {self.params.placeholderToken}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

        # Convert the initializer_token to ids
        token_ids = tokenizer.encode(self.params.initializerToken, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")      # TODO use all tokens in the initializer instead of erroring
        initializer_token_id = token_ids[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        with torch.no_grad():
            for token_id in placeholder_token_ids:
                token_embeds[token_id] = token_embeds[initializer_token_id].clone()

        # Freeze vae and unet
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if self.params.gradientCheckpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            unet.train()
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        if self.params.enableXformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.params.scaleLearningRate:
            self.params.learningRate = (
                self.params.learningRate * self.params.gradientAccumulationSteps * self.params.batchSize * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=self.params.learningRate,
            betas=(self.params.adamBeta1, self.params.adamBeta2),
            weight_decay=self.params.adamWeightDecay,
            eps=self.params.adamEpsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = TextualInversionDataset(
            data_root=self.params.trainDataDir,
            tokenizer=tokenizer,
            size=self.params.resolution,
            placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
            repeats=self.params.repeats,
            learnable_property=self.params.learnableProperty,
            center_crop=self.params.centreCrop,
            set="train",
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params.batchSize, shuffle=True, num_workers=0
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.params.gradientAccumulationSteps)
        if self.params.maxSteps is None:
            self.params.maxSteps = self.params.numEpochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.params.learningRateSchedule,
            optimizer=optimizer,
            num_warmup_steps=self.params.learningRateWarmupSteps * self.accelerator.num_processes,
            num_training_steps=self.params.maxSteps * self.accelerator.num_processes,
            num_cycles=self.params.learningRateNumCycles,
        )

        # Prepare everything with our `accelerator`.
        text_encoder, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device and cast to weight_dtype
        unet.to(self.accelerator.device, dtype=weight_dtype)
        vae.to(self.accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / params.gradientAccumulationSteps)
        if overrode_max_train_steps:
            self.params.maxSteps = self.params.numEpochs * num_update_steps_per_epoch
        self.params.numEpochs = math.ceil(self.params.maxSteps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("textual_inversion", config=vars(params))

        # Train!
        total_batch_size = self.params.batchSize * self.accelerator.num_processes * self.params.gradientAccumulationSteps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self.params.numEpochs}")
        logger.info(f"  Instantaneous batch size per device = {self.params.batchSize}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.params.gradientAccumulationSteps}")
        logger.info(f"  Total optimization steps = {self.params.maxSteps}")
        global_step = 0
        first_epoch = 0
    
        initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.params.maxSteps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        # keep original embeddings as reference
        orig_embeds_params = self.accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

        for epoch in range(first_epoch, self.params.numEpochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    self.accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                    with torch.no_grad():
                        self.accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % self.params.saveSteps == 0:
                        self.save_progress(global_step, text_encoder, placeholder_token_ids)

                    if self.accelerator.is_main_process:
                        if self.params.validationPrompt is not None and global_step % self.params.validationSteps == 0:
                            self.log_validation(global_step, text_encoder, tokenizer, unet, vae, weight_dtype, epoch)

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.params.maxSteps:
                    break

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:       
            # Save the newly trained embeddings
            self.save_progress(global_step, text_encoder, placeholder_token_ids)

        self.accelerator.end_training()


    def log_validation(self, global_step, text_encoder, tokenizer, unet, vae, weight_dtype, epoch):
        logger.info(
            f"Running validation for step {global_step}... \n Generating {self.params.numValidtionImages} images with prompt:"
            f" {self.params.validationPrompt}."
        )
        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            self.params.model,
            text_encoder=self.accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = None if self.params.validationSeed is None else torch.Generator(device=self.accelerator.device).manual_seed(self.params.validationSeed)
        images = []
        for _ in range(self.params.numValidtionImages):
            # with torch.autocast("cuda"):
            image = pipeline(params.validationPrompt, num_inference_steps=25, generator=generator).images[0]
            display(image)
            images.append(image)

        del pipeline
        # torch.cuda.empty_cache()
        return images


    def save_progress(self, global_step, text_encoder, placeholder_token_ids):
        logger.info("Saving embeddings")

        weight_name = (
            f"learned_embeds-steps-{global_step}.safetensors"
            if self.params.safetensors
            else f"learned_embeds-steps-{global_step}.bin"
        )
        save_path = os.path.join(self.params.outputDir, weight_name)

        learned_embeds = (
            self.accelerator.unwrap_model(text_encoder)
            .get_input_embeddings()
            .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
        )
        learned_embeds_dict = {self.params.placeholderToken: learned_embeds.detach().cpu()}

        if self.params.safetensors:
            safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, save_path)
