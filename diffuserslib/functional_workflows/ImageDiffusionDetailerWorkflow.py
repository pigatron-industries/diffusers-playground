from diffuserslib.functional import *


#TODO this is a work in progress
class ImageDiffusionDetailerWorkflow(WorkflowBuilder):
    """ Image diffusion with a set of fixed conditioning models for adding detail to a large image. """

    def __init__(self):
        super().__init__("Image Diffusion - Detailer", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        initimage_scale_input = FloatUserInputNode(value = 0.2, name = "initimage_scale")
        canny_scale_input = FloatUserInputNode(value = 0.9, name = "canny_scale")
        ipadapter_scale_input = FloatUserInputNode(value = 0.6, name = "ipadapter_scale")

        model_input = DiffusionModelUserInputNode(name = "model")
        lora_input = LORAModelUserInputNode(diffusion_model_input = model_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name = "prompt")
        negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        seed_input = SeedUserInputNode(value = None, name = "seed")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                                name = "scheduler",
                                                options = ImageDiffusionNode.SCHEDULERS)
        
        # TODO work out how to do tiling
        tiled_input = TiledImageInputNode(name = "tile_input")
        
        # preprocessing
        canny_image_input = CannyEdgeDetectionNode(image = tiled_input, name = "canny_image")
        initimage_input = tiled_input
        ipadapter_input = tiled_input

        # conditioning
        conditioning_inputs = [
            ConditioningInputNode(image = initimage_input, model = "initimage", scale = initimage_scale_input, name = "initimage_input"),
            ConditioningInputNode(image = canny_image_input, model = "canny", scale = canny_scale_input, name = "canny_input"),
            ConditioningInputNode(image = ipadapter_input, model = "ipadapter", scale = ipadapter_scale_input, name = "ipadapter_input")
        ]
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")

        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = conditioning_inputs)
        
        tiled_output = TiledImageOutputNode(image = diffusion, name = "tile_output")
        
        model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(model_input.basemodel)))
        return diffusion
