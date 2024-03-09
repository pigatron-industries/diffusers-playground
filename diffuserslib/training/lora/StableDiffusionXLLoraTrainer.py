from .LoraTrainingParameters import LoraTrainingParameters
from ..DiffusersTrainer import DiffusersTrainer
from .DreamBoothDataset import DreamBoothDataset

import os
import gc
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import torch.nn.functional as F
from IPython.display import display
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)

from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from peft.utils import get_peft_model_state_dict
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
        if not self.params.trainTextEncoder and not self.train_dataset.custom_instance_prompts:
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = self.compute_text_embeddings(self.params.instancePrompt)

        # Handle class prompt for prior-preservation.
        if self.params.priorPreservation and not self.params.trainTextEncoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = self.compute_text_embeddings(self.params.classPrompt)

        # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
        # pack the statically computed variables appropriately here. This is so that we don't
        # have to pass them to the dataloader.
        if not self.train_dataset.custom_instance_prompts:
            if not self.params.trainTextEncoder:
                prompt_embeds = instance_prompt_hidden_states
                unet_add_text_embeds = instance_pooled_prompt_embeds
                if self.params.priorPreservation:
                    prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                    unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
            # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
            # batch prompts on all training steps
            else:
                text_encoder_tokens = []
                for text_encoder_trainer in self.text_encoder_trainers:
                    tokens = text_encoder_trainer.tokenize_prompt(self.params.instancePrompt)
                    if self.params.priorPreservation:
                        class_tokens = text_encoder_trainer.tokenize_prompt(self.params.classPrompt)
                        tokens = torch.cat([tokens, class_tokens], dim=0)
                    text_encoder_tokens.append(tokens)

        self.calcTrainingSteps()
        self.createScheduler()

        # Prepare everything with our `accelerator`.
        self.optimizer, self.train_dataloader, self.lr_scheduler, self.unet = self.accelerator.prepare(self.optimizer, self.train_dataloader, self.lr_scheduler, self.unet)
        if(self.params.trainTextEncoder):
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder_trainer.text_encoder = self.accelerator.prepare(text_encoder_trainer.text_encoder)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.calcTrainingSteps()

        # Train!
        total_batch_size = self.params.batchSize * self.accelerator.num_processes * self.params.gradientAccumulationSteps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.params.numEpochs}")
        logger.info(f"  Instantaneous batch size per device = {self.params.batchSize}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.params.gradientAccumulationSteps}")
        logger.info(f"  Total optimization steps = {self.params.maxSteps}")
        self.global_step = 0
        self.start_epoch = 0
        self.first_epoch = 0

        self.createProgressBar()
        # TODO log validation images
        self.train_loop()


    def generate_class_images(self):
        # TODO
        pass


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
    

    def compute_text_embeddings(self, prompt):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=prompt)
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds


    def save_params(self):
        if self.params.outputDir is not None:
            os.makedirs(self.params.outputDir, exist_ok=True)
        trainparamsfile = f"{self.params.outputDir}/{self.params.outputPrefix}-params.json"
        with open(trainparamsfile, 'w') as f:
            f.write(self.params.toJson())


    def train_loop(self):
        for epoch in range(self.first_epoch, self.params.numEpochs):
            self.unet.train()
            if self.params.trainTextEncoder:
                for text_encoder_trainer in self.text_encoder_trainers:
                    text_encoder_trainer.text_encoder.train()
                    self.accelerator.unwrap_model(text_encoder_trainer.text_encoder).text_model.embeddings.requires_grad_(True)

            for step, batchitem in enumerate(self.train_dataloader):
                loss = self.train_step(batchitem)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if (self.accelerator.is_main_process and self.accelerator.sync_gradients):
                    self.progress_bar.update(1)
                    self.global_step += 1
                    if self.global_step % self.params.saveSteps == 0:
                        self.save_progress()
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


    def train_step(self, batchitem):
        with self.accelerator.accumulate(self.unet):
            pixel_values = batchitem["pixel_values"].to(dtype=self.vae.dtype)
            prompts = batchitem["prompts"]

            # encode batch prompts when custom prompts are provided for each image -
            if self.train_dataset.custom_instance_prompts:
                if not self.params.trainTextEncoder:
                    prompt_embeds, unet_add_text_embeds = self.compute_text_embeddings(prompts)
                else:
                    self.text_encoder_tokens = []
                    for text_encoder_trainer in self.text_encoder_trainers:
                        tokens = text_encoder_trainer.tokenize_prompt(prompts)
                        self.text_encoder_tokens.append(tokens)

            # Convert images to latent space
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
            model_input = model_input.to(self.weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            timesteps = timesteps.long()

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

            # time ids
            add_time_ids = torch.cat(
                [
                    self.compute_time_ids(original_size=s, crops_coords_top_left=c)
                    for s, c in zip(batchitem["original_sizes"], batchitem["crop_top_lefts"])
                ]
            )

            # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
            if not self.train_dataset.custom_instance_prompts:
                elems_to_repeat_text_embeds = bsz // 2 if self.params.priorPreservation else bsz
            else:
                elems_to_repeat_text_embeds = 1

            # Predict the noise residual
            if not self.params.trainTextEncoder:
                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                }
                prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                model_pred = self.unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]
            else:
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt=None, text_input_ids_list=self.text_encoder_tokens)
                unet_added_conditions.update(
                    {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                )
                prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                model_pred = self.unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            
            if self.params.priorPreservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Compute loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if self.params.priorPreservation:
                # Add the prior loss to the instance loss.
                loss = loss + self.params.priorLossWeight * prior_loss

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = [self.unet_lora_parameters]
                if self.params.trainTextEncoder:
                    params_to_clip.extend([text_encoder_trainer.params for text_encoder_trainer in self.text_encoder_trainers])
                self.accelerator.clip_grad_norm_(params_to_clip, self.params.maxGradientNorm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            return loss



    def compute_time_ids(self, original_size, crops_coords_top_left):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = self.params.resolution
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(self.accelerator.device, dtype=self.weight_dtype)
        return add_time_ids
    

    def encode_prompt(self, prompt, text_input_ids_list=None):
        prompt_embeds_list = []

        for i, text_encoder_trainer in enumerate(self.text_encoder_trainers):
            if(text_input_ids_list is not None):
                text_input_ids = text_input_ids_list[i]
            else:
                text_input_ids = text_encoder_trainer.tokenize_prompt(prompt)
            prompt_embeds = text_encoder_trainer.text_encoder(text_input_ids.to(text_encoder_trainer.text_encoder.device), output_hidden_states=True, return_dict=False)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds[-1][-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds
    

    def load_validation_pipeline(self):
        if(self.params.numValidationImages > 0):
            self.validationPipeline = DiffusionPipeline.from_pretrained(self.params.validationModel, safety_checker=None, torch_dtype=self.weight_dtype)
            self.validationPipeline.scheduler = EulerDiscreteScheduler.from_config(self.validationPipeline.scheduler.config)
            self.validationPipeline = self.validationPipeline.to(self.accelerator.device)


    def update_validation_pipeline(self):
        # self.validationPipeline.text_encoder = self.accelerator.unwrap_model(self.text_encoder_trainers[0].text_encoder)
        # self.validationPipeline.tokenizer = self.text_encoder_trainers[0].tokenizer
        # self.validationPipeline.text_encoder_2 = self.accelerator.unwrap_model(self.text_encoder_trainers[1].text_encoder)
        # self.validationPipeline.tokenizer_2 = self.text_encoder_trainers[1].tokenizer
        self.validationPipeline.unet = self.accelerator.unwrap_model(self.unet)


    def log_validation(self, epoch):
        logger.info(
            f"Running validation for step {self.global_step}... \n Generating {self.params.numValidationImages} images with prompt:"
            f" {self.params.validationPrompt}."
        )

        gc.collect()
        torch.mps.empty_cache()
        torch.cuda.empty_cache()

        self.update_validation_pipeline()

        # run inference
        generator = None if self.params.validationSeed is None else torch.Generator(device=self.accelerator.device).manual_seed(self.params.validationSeed)
        images = []
        for i in range(self.params.numValidationImages):
            image = self.validationPipeline(self.params.validationPrompt, 
                             negative_prompt = self.params.validationNegativePrompt,
                             num_inference_steps = self.params.validationInferenceSteps, 
                             guidance_scale=9.0,
                             generator = generator,
                             width = self.params.validationSize[0], 
                             height = self.params.validationSize[1]).images[0]
            display(image)
            images.append(image)
            filename = f"{self.params.loraName}-{self.global_step}-{i}"
            image.save(f"{self.params.outputDir}/{filename}.png")

        gc.collect()
        torch.mps.empty_cache()
        torch.cuda.empty_cache()

        return images


    def save_progress(self):
        # Save the lora layers
        unet = self.accelerator.unwrap_model(self.unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        text_encoder_lora_layers = []
        if self.params.trainTextEncoder:
            for text_encoder_trainer in self.text_encoder_trainers:
                text_encoder = self.accelerator.unwrap_model(text_encoder_trainer.text_encoder).to(torch.float32)
                text_encoder_lora_layers.append(convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder)))

        self.pipeline.save_lora_weights(
            save_directory = self.params.outputDir,
            weight_name = f"{self.params.loraName}-{self.global_step}",
            unet_lora_layers = unet_lora_layers,
            text_encoder_lora_layers = text_encoder_lora_layers[0] if self.params.trainTextEncoder else None,
            text_encoder_2_lora_layers = text_encoder_lora_layers[1] if self.params.trainTextEncoder else None,
        )

        # save_path = os.path.join(self.params.outputDir, f"checkpoint-{self.global_step}")
        # self.accelerator.save_state(save_path)
        # logger.info(f"Saved state to {save_path}")