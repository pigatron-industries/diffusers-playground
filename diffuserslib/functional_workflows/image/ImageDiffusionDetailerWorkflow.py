from diffuserslib.functional.WorkflowBuilder import WorkflowBuilder
from diffuserslib.functional.nodes.user import *
from diffuserslib.functional.nodes.image.diffusers import *
from diffuserslib.functional.nodes.image.process import *


#TODO this is a work in progress
class ImageDiffusionDetailerWorkflow(WorkflowBuilder):
    """ Image diffusion with a set of fixed conditioning models for adding detail to a large image. """

    def __init__(self):
        super().__init__("Image Diffusion - Detailer", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        image_input = ImageUploadInputNode(name = "image")
        initimage_scale_input = FloatUserInputNode(value = 0.2, name = "initimage_scale")

        canny_scale_input = FloatUserInputNode(value = 0.9, name = "canny_scale")
        canny_model_input = ListSelectUserInputNode(options = [], value = "", name = "canny_model")

        ipadapter_scale_input = FloatUserInputNode(value = 0, name = "ipadapter_scale")
        ipadapter_model_input = ListSelectUserInputNode(options = [], value = "", name = "ipadapter_model")

        model_input = DiffusionModelUserInputNode(name = "model")
        lora_input = LORAModelUserInputNode(diffusion_model_input = model_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name = "prompt")
        negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        seed_input = SeedUserInputNode(value = None, name = "seed")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                                name = "scheduler",
                                                options = ImageDiffusionNode.SCHEDULERS)
        
        # preprocessing
        canny_image = CannyEdgeDetectionNode(image = image_input, name = "canny_image")

        # conditioning inputs
        initimage_condition = ConditioningInputNode(image = image_input, model = "initimage", scale = initimage_scale_input, name = "initimage_condition")
        cannyimage_condition = ConditioningInputNode(image = canny_image, model = canny_model_input, type="controlnet", scale = canny_scale_input, name = "canny_condition")
        ipadapter_condition = ConditioningInputNode(image = image_input, model = ipadapter_model_input, scale = ipadapter_scale_input, name = "ipadapter_condition")

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")

        # On model update
        def modelUpdate():
            assert DiffusersPipelines.pipelines is not None, "DiffusersPipelines is not initialized"
            basemodel = model_input.basemodel
            prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(basemodel))
            controlnet_models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("controlnet", basemodel)
            canny_model_input.setOptions([model for model in controlnet_models if "canny" in model])
            ipadapter_models = DiffusersPipelines.pipelines.presets.getModelsByTypeAndBase("ipadapter", basemodel)
            ipadapter_model_input.setOptions([model for model in ipadapter_models])
        model_input.addUpdateListener(modelUpdate)

        diffusion = ImageDiffusionTiledNode(models = model_input,
                                    loras = lora_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = [initimage_condition, cannyimage_condition, ipadapter_condition])
        
        return diffusion
