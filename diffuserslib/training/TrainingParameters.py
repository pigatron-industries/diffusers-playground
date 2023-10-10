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
    numVectors: int = 1                 # Number of vectors to train.

    batchSize: int = 16                 # Batch size (per device) for the training dataloader.
    saveSteps: int = 500                # Save a checkpoint of the training state every X updates.
    maxSteps: int = 1000                # Total number of training steps to perform.
    gradientAccumulationSteps: int = 1  # Number of updates steps to accumulate before performing a backward/update pass.
    gradientCheckpointing: bool = False # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
    initialLearningRate: float = 1e-4   # Initial learning rate (after the potential warmup period) to use.
    scaleLearningRate: bool = True      # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
    learningRateSchedule: str = 'constant' # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    learningRateWarmupSteps: int = 500  # Number of steps for the warmup in the lr scheduler.
    learningRateNumCycles: int = 1      # Number of hard resets of the lr in cosine_with_restarts scheduler.




#     parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
#     parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
#     parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
#     parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")


#     parser.add_argument(
#         "--validation_prompt",
#         type=str,
#         default=None,
#         help="A prompt that is used during validation to verify that the model is learning.",
#     )
#     parser.add_argument(
#         "--num_validation_images",
#         type=int,
#         default=4,
#         help="Number of images that should be generated during validation with `validation_prompt`.",
#     )
#     parser.add_argument(
#         "--validation_steps",
#         type=int,
#         default=100,
#         help=(
#             "Run validation every X steps. Validation consists of running the prompt"
#             " `args.validation_prompt` multiple times: `args.num_validation_images`"
#             " and logging the images."
#         ),
#     )
#     parser.add_argument(
#         "--validation_epochs",
#         type=int,
#         default=None,
#         help=(
#             "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
#             " `args.validation_prompt` multiple times: `args.num_validation_images`"
#             " and logging the images."
#         ),
#     )