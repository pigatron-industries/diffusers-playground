from dataclasses import dataclass, field

@dataclass
class TrainingParameters:
    model: str = 'runwayml/stable-diffusion-v1-5'
    trainDataDir: str = '/train'        # A folder containing the training data.
    outputDir: str = '/output'          # A folder where the checkpoints will be saved.
    placeholderToken: str = '<token>'   # A token to use as a placeholder for the concept.
    initializerToken: str = 'person'    # A token to use as initializer word.
    learnableProperty: str = 'object'   # Choose between 'object' and 'style'
    seed: int|None = None               # A seed for reproducible training.
    resolution: int = 512               # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
    centreCrop: bool = False            # Whether to center crop images before resizing to resolution.
    repeats: int = 100                  # Number of times to repeat the dataset.
    numVectors: int = 1                 # Number of vectors to train.
    safetensors: bool = True            # Whether to save in savetensor format.

    validationPrompt: str = ''          # A prompt that is used during validation to verify that the model is learning.
    validationSteps: int = 100          # Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.
    numValidtionImages: int = 4         # Number of images that should be generated during validation with `validation_prompt`.

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
    adamBeta1: float = 0.9              # The beta1 parameter for the Adam optimizer.
    adamBeta2: float = 0.999            # The beta2 parameter for the Adam optimizer.
    adamWeightDecay: float = 1e-2       # Weight decay to use.
    adamEpsilon: float = 1e-08          # Epsilon value for the Adam optimizer
