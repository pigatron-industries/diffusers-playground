from .TrainingParameters import TrainingParameters
from .TextEncoderTrainer import TextEncoderTrainer

import torch

import logging
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DiffusionPipeline

logger = get_logger(__name__)


class DiffusersTrainer():

    def __init__(self, params:TrainingParameters):
        self.params = params
        self.accelerator = Accelerator(
            gradient_accumulation_steps = params.gradientAccumulationSteps,
            project_config = ProjectConfiguration(project_dir=params.outputDir),
            mixed_precision = params.mixedPrecision
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


    def load_models(self):
        self.pipeline = DiffusionPipeline.from_pretrained(self.params.model, safety_checker=None, torch_dtype=self.weight_dtype)
        self.noise_scheduler = self.pipeline.components["scheduler"]
        self.vae = self.pipeline.components["vae"]
        self.unet = self.pipeline.components["unet"]
        self.text_encoder_trainers = [self.createTextEncoderTrainer("tokenizer", "text_encoder")]


    def createTextEncoderTrainer(self, tokenizer_subfolder:str, text_encoder_subfolder:str):
        return TextEncoderTrainer(
            self.pipeline.components[tokenizer_subfolder],
            self.pipeline.components[text_encoder_subfolder],
            self.accelerator)
    

    def scaleLearningRate(self):
        if self.params.scaleLearningRate:
            self.params.learningRate = (
                self.params.learningRate * self.params.gradientAccumulationSteps * self.params.batchSize * self.accelerator.num_processes
            )