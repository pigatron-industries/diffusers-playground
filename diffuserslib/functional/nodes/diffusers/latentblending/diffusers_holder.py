import torch
import numpy as np
import warnings

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .utils import interpolate_spherical
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)


class DiffusersHolder():
    def __init__(self, pipe):
        # Base settings
        self.negative_prompt = ""
        self.guidance_scale = 5.0
        self.num_inference_steps = 30

        # Check if valid pipe
        self.pipe = pipe
        self.device = str(pipe._execution_device)
        self.init_types()

        self.width_latent = self.pipe.unet.config.sample_size
        self.height_latent = self.pipe.unet.config.sample_size
        self.width_img = self.width_latent  * self.pipe.vae_scale_factor
        self.height_img = self.height_latent  * self.pipe.vae_scale_factor
        

    def init_types(self):
        assert hasattr(self.pipe, "__class__"), "No valid diffusers pipeline found."
        assert hasattr(self.pipe.__class__, "__name__"), "No valid diffusers pipeline found."
        if self.pipe.__class__.__name__ == 'StableDiffusionXLPipeline':
            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
            prompt_embeds, _, _, _ = self.pipe.encode_prompt("test")
        else:
            prompt_embeds = self.pipe._encode_prompt("test", self.device, 1, True)
        self.dtype = prompt_embeds.dtype
        
        self.is_sdxl_turbo = 'turbo' in self.pipe._name_or_path
        

    def set_num_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    def set_dimensions(self, size_output):
        s = self.pipe.vae_scale_factor
        if size_output is None:
            width = self.pipe.unet.config.sample_size
            height = self.pipe.unet.config.sample_size
        else:
            width, height = size_output
        self.width_img = int(round(width / s) * s)
        self.width_latent = int(self.width_img / s)
        self.height_img = int(round(height / s) * s)
        self.height_latent = int(self.height_img / s)
        print(f"set_dimensions to width={width} and height={height}")

    def set_negative_prompt(self, negative_prompt):
        r"""Set the negative prompt. Currenty only one negative prompt is supported
        """
        if isinstance(negative_prompt, str):
            self.negative_prompt = [negative_prompt]
        else:
            self.negative_prompt = negative_prompt

        if len(self.negative_prompt) > 1:
            self.negative_prompt = [self.negative_prompt[0]]

    def get_text_embedding(self, prompt):
        do_classifier_free_guidance = self.guidance_scale > 1 and self.pipe.unet.config.time_cond_proj_dim is None
        text_embeddings = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=self.negative_prompt,
            negative_prompt_2=self.negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,#self.pipe._clip_skip,
        )
        return text_embeddings

    def get_noise(self, seed=420):
        
        latents = self.pipe.prepare_latents(
            1,
            self.pipe.unet.config.in_channels,
            self.height_img,
            self.width_img,
            torch.float32,
            self.pipe._execution_device,
            torch.Generator(device=self.device).manual_seed(int(seed)),
            None,
        )
        
        return latents


    @torch.no_grad()
    def latent2image(
            self,
            latents: torch.FloatTensor,
            output_type="pil"):
        r"""
        Returns an image provided a latent representation from diffusion.
        Args:
            latents: torch.FloatTensor
                Result of the diffusion process.
            output_type: "pil" or "np"
        """
        assert output_type in ["pil", "np"]
            
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
    
        if needs_upcasting:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)
    
        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
    
        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float16)
    
        image = self.pipe.image_processor.postprocess(image, output_type=output_type)[0]
        
        return image
        

    def prepare_mixing(self, mixing_coeffs, list_latents_mixing):
        if type(mixing_coeffs) == float:
            list_mixing_coeffs = (1 + self.num_inference_steps) * [mixing_coeffs]
        elif type(mixing_coeffs) == list:
            assert len(mixing_coeffs) == self.num_inference_steps, f"len(mixing_coeffs) {len(mixing_coeffs)} != self.num_inference_steps {self.num_inference_steps}"
            list_mixing_coeffs = mixing_coeffs
        else:
            raise ValueError("mixing_coeffs should be float or list with len=num_inference_steps")
        if np.sum(list_mixing_coeffs) > 0:
            assert len(list_latents_mixing) == self.num_inference_steps, f"len(list_latents_mixing) {len(list_latents_mixing)} != self.num_inference_steps {self.num_inference_steps}"
        return list_mixing_coeffs

    @torch.no_grad()
    def run_diffusion(
            self,
            text_embeddings: torch.FloatTensor,
            latents_start: torch.FloatTensor,
            idx_start: int = 0,
            list_latents_mixing=None,
            mixing_coeffs=0.0,
            return_image: Optional[bool] = False):

        return self.run_diffusion_sd_xl(text_embeddings, latents_start, idx_start, list_latents_mixing, mixing_coeffs, return_image)



    @torch.no_grad()
    def run_diffusion_sd_xl(
        self,
        text_embeddings: tuple,
        latents_start: torch.FloatTensor,
        idx_start: int = 0,
        list_latents_mixing=None,
        mixing_coeffs=0.0,
        return_image: Optional[bool] = False,
    ):
        
        
        prompt_2 = None
        height = None
        width = None
        timesteps = None
        denoising_end = None
        negative_prompt_2 = None
        num_images_per_prompt = 1
        eta = 0.0
        generator = None
        latents = None
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        ip_adapter_image = None
        output_type = "pil"
        return_dict = True
        cross_attention_kwargs = None
        guidance_rescale = 0.0
        original_size = None
        crops_coords_top_left = (0, 0)
        target_size = None
        negative_original_size = None
        negative_crops_coords_top_left = (0, 0)
        negative_target_size = None
        clip_skip = None
        callback = None
        callback_on_step_end = None
        callback_on_step_end_tensor_inputs = ["latents"]
        # kwargs are additional keyword arguments and don't need a default value set here.

        # 0. Default height and width to unet
        height = height or self.pipe.default_sample_size * self.pipe.vae_scale_factor
        width = width or self.pipe.default_sample_size * self.pipe.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. skipped.

        self.pipe._guidance_scale = self.guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._denoising_end = denoising_end
        self.pipe._interrupt = False

        # 2. Define call parameters
        list_mixing_coeffs = self.prepare_mixing(mixing_coeffs, list_latents_mixing)
        batch_size = 1

        device = self.pipe._execution_device

        # 3. Encode input prompt
        lora_scale = None
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipe.scheduler, self.num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = latents_start.clone()
        list_latents_out = []

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            output_hidden_state = False if isinstance(self.pipe.unet.encoder_hid_proj, ImageProjection) else True
            image_embeds, negative_image_embeds = self.pipe.encode_image(
                ip_adapter_image, device, num_images_per_prompt, output_hidden_state
            )
            if self.pipe.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipe.scheduler.order, 0)

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
            
        self.pipe._num_timesteps = len(timesteps)
        for i, t in enumerate(timesteps):
            # Set the right starting latents
            # Write latents out and skip
            if i < idx_start:
                list_latents_out.append(None)
                continue
            elif i == idx_start:
                latents = latents_start.clone()
                
            # Mix latents for crossfeeding
            if i > 0 and list_mixing_coeffs[i] > 0:
                latents_mixtarget = list_latents_mixing[i - 1].clone()
                latents = interpolate_spherical(latents, latents_mixtarget, list_mixing_coeffs[i])


            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.pipe.do_classifier_free_guidance else latents

            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            if ip_adapter_image is not None:
                added_cond_kwargs["image_embeds"] = image_embeds
            noise_pred = self.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.pipe.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.pipe.do_classifier_free_guidance and self.pipe.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # Append latents
            list_latents_out.append(latents.clone())
                
                

        if return_image:
            return self.latent2image(latents)
        else:
            return list_latents_out

