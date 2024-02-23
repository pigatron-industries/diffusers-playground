from diffuserslib.functional import *


class ImageDiffusionConditioningWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion", Image.Image, workflow=True, subworkflow=False)


    def build(self):
        model_input = DiffusionModelUserInputNode(name = "model")
        lora_input = LORAModelUserInputNode(diffusion_model_input = model_input, name = "lora")
        size_input = SizeUserInputNode(value = (512, 512), name = "size")
        prompt_input = TextAreaInputNode(value = "", name = "prompt")
        negprompt_input = TextAreaInputNode(value = "", name = "negprompt")
        steps_input = IntUserInputNode(value = 40, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        seed_input = IntUserInputNode(value = None, name = "seed")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", 
                                                name = "scheduler",
                                                options = ImageDiffusionNode.SCHEDULERS)
        
        def conditioning_input():
            conditioning_model_input = ConditioningModelUserInputNode(diffusion_model_input = model_input, name = "model")
            scale_input = FloatUserInputNode(value = 1.0, name = "scale")
            image_input = ImageSelectInputNode(name = "image")
            return ConditioningInputNode(image = image_input, model = conditioning_model_input, scale = scale_input, name = "conditioning_input")

        conditioning_inputs = ListUserInputNode(input_node_generator = conditioning_input, name = "conditioning_inputs")
        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")

        diffusion = ImageDiffusionNode(models = model_input,
                                    loras = lora_input,
                                    size = size_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = conditioning_inputs)
        
        model_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(model_input.basemodel)))
        return diffusion
