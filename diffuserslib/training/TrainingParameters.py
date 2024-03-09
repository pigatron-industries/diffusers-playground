from dataclasses import dataclass, asdict, field
from typing import Tuple, List
import json

@dataclass
class TrainingParameters:
    base: str
    description: str = ''
    name: str = 'lora'              # The name of the training run.
    model: str = 'runwayml/stable-diffusion-v1-5'
    trainDataDir: str = '/train'        # A folder containing the training data.
    trainDataFiles: List[str] = field(default_factory=lambda: ["*"])
    outputDir: str = '/output'          # A folder where the checkpoints will be saved.
    outputPrefix: str = 'object'        # The filename of the checkpoint.
    seed: int|None = None               # A seed for reproducible training.
    resolution: Tuple[int, int]|int = (512, 512) # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
    centreCrop: bool = False            # Whether to center crop images before resizing to resolution.
    repeats: int = 100                  # Number of times to repeat the dataset.
    mixedPrecision: str = 'no'          # Whether to use mixed precision training. Choose between ['no', 'fp16', 'bf16']
    safetensors: bool = True            # Whether to save in savetensor format.

    numValidationImages: int = 4        # Number of images that should be generated during validation with `validation_prompt`.
    validationSize: Tuple[int, int] = (512, 512) # The resolution for validation images.
    validationSteps: int = 100          # Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.
    validationModel: str = 'runwayml/stable-diffusion-v1-5' # A path to a model that should be used for validation.
    validationPrompt: str = ''          # A prompt that is used during validation to verify that the model is learning.
    validationNegativePrompt: str = ''  # A negative prompt that is used during validation to verify that the model is learning.
    validationSeed: int = 0             # Seed to use for validation images.
    validationInferenceSteps: int = 40  # Number of inference steps for validation image.

    batchSize: int = 16                 # Batch size (per device) for the training dataloader.
    saveSteps: int = 500                # Save a checkpoint of the training state every X updates.
    maxSteps: int = 1000                # Total number of training steps to perform.
    numEpochs: int = 100                # Number of epochs to train for.
    gradientAccumulationSteps: int = 1  # Number of updates steps to accumulate before performing a backward/update pass.
    gradientCheckpointing: bool = False # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    enableXformers: bool = False        # Whether or not to use Xformers for the training.
    learningRate: float = 1e-4          # Initial learning rate (after the potential warmup period) to use.
    scaleLearningRate: bool = True      # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    learningRateSchedule: str = 'constant' # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    learningRateWarmupSteps: int = 500  # Number of steps for the warmup in the lr scheduler.
    learningRateNumCycles: int = 1      # Number of hard resets of the lr in cosine_with_restarts scheduler.
    learningRatePower: float = 1.0      # Power factor for polynomial learning rate schedule.
    adamBeta1: float = 0.9              # The beta1 parameter for the Adam optimizer.
    adamBeta2: float = 0.999            # The beta2 parameter for the Adam optimizer.
    adamWeightDecay: float = 1e-2       # Weight decay to use.
    adamEpsilon: float = 1e-08          # Epsilon value for the Adam optimizer


    def __init__(self, base:str, outputDirPrefix:str=None, **kwargs):
        self.base = base
        if base == 'sd_1_5':
            self.model = 'runwayml/stable-diffusion-v1-5'
        elif base == 'sd_2_1':
            self.model = 'stabilityai/stable-diffusion-2-1'
        elif base == 'sdxl_1_0':
            self.model = 'stabilityai/stable-diffusion-xl-base-1.0'
        self.validationModel = self.model

        for k,v in kwargs.items():
            setattr(self, k, v)

        self.validationSteps = self.saveSteps
        if outputDirPrefix is not None:
            self.outputDir = f"{outputDirPrefix}/{self.base}/{self.name}"


    def toJson(self):
        return json.dumps(self, cls=DataclassEncoder, indent=2)
    

class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (TrainingParameters)):
            return asdict(obj)
        return super().default(obj)