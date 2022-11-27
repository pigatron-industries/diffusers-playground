import os
import random
import math
import PIL
import torch
import numpy as np
import accelerate
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

DEFAULT_BASE_MODEL = 'runwayml/stable-diffusion-v1-5'
DEFAULT_DATA_DIR = '/content/data'
DEFAULT_OUT_DIR = '/content/out'
INPUT_DIR = '/input'
PIPE_DIR = '/pipe'

imagenet_object_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class EmbedType:
    object = 1
    style = 2





class TextEmbeddingTrainer():

    def __init__(self, model=DEFAULT_BASE_MODEL, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
        self.model = model
        self.out_dir = out_dir
        self.data_dir = data_dir
        self.input_dir = data_dir + INPUT_DIR
        self.pipe_dir = data_dir + PIPE_DIR
        os.makedirs(self.out_dir, exists_ok=True)
        os.makedirs(self.pipe_dir, exists_ok=True)
        os.makedirs(self.input_dir, exists_ok=True)
    

    def trainSetup(self, embed_type, train_token, init_token):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.model, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.model, subfolder="unet")
        self.train_token = train_token

        num_added_tokens = self.tokenizer.add_tokens(train_token)
        if (num_added_tokens == 0):
            raise ValueError(f"The tokenizer already contains the token {train_token}.")

        token_ids = self.tokenizer.encode(init_token, add_special_tokens=False)
        if (len(token_ids) > 1):
            raise ValueError("The initializer token must be a single token.")

        self.init_token_id = token_ids[0]
        self.train_token_id = self.tokenizer.convert_tokens_to_ids(train_token)

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        token_embeds = self.text_encoder.get_input_embeddings().weight.data
        token_embeds[self.train_token_id] = token_embeds[self.init_token_id]

        self.freezeParams(self.vae.parameters())
        self.freezeParams(self.unet.parameters())
        self.freezeParams(self.text_encoder.text_model.encoder.parameters())
        self.freezeParams(self.text_encoder.text_model.final_layer_norm.parameters())
        self.freezeParams(self.text_encoder.text_model.embeddings.position_embedding.parameters())

        self.train_dataset = TextualInversionDataset(data_root=self.input_dir, tokenizer=self.tokenizer, size=512,
            train_token=train_token,repeats=100,learnable_property=embed_type, center_crop=False,set="train")


    def freezeParams(self, params):
        for param in params:
            param.requires_grad = False


    def train(self, train_steps=1000, learning_rate=5e-04, train_batch_size=1):
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = 4
        self.scale_lr = True
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, tensor_format="pt")
        accelerate.notebook_launcher(self.training_function, args=(self))


    def training_function(self):
        logger = get_logger(__name__)
        accelerator = Accelerator(gradient_accumulation_steps=self.gradient_accumulation_steps)
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        if self.scale_lr:
            self.learning_rate = (self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes)

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=self.learning_rate)
        text_encoder, optimizer, train_dataloader = accelerator.prepare(self.text_encoder, optimizer, train_dataloader)

        self.vae.to(accelerator.device)
        self.unet.to(accelerator.device)
        self.vae.eval()
        self.unet.eval()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.train_steps / num_update_steps_per_epoch)
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.train_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(self.tokenizer)) != self.train_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= self.train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=self.vae,
                unet=self.unet,
                tokenizer=self.tokenizer,
                scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
                safety_checker=None
            )
            pipeline.save_pretrained(self.pipe_dir)
            # Also save the newly trained embeddings
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[self.train_token_id]
            learned_embeds_dict = {self.train_token: learned_embeds.detach().cpu()}
            torch.save(learned_embeds_dict, os.path.join(self.out_dir, f"embed_{self.train_token.strip('<>')}.bin"))



class TextualInversionDataset(Dataset):
    def __init__(self, data_root, tokenizer, learnable_property=EmbedType.object, size=512, repeats=100, interpolation=PIL.Image.LANCZOS, 
                 flip_p=0.5, set="train", train_token="*", center_crop=False):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.train_token = train_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        if set == "train":
            self._length = self.num_images * repeats
        self.interpolation = interpolation
        self.templates = imagenet_style_templates_small if learnable_property == EmbedType.style else imagenet_object_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        train_token = self.train_token
        text = random.choice(self.templates).format(train_token)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
