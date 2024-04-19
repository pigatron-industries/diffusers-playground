from diffuserslib.functional import *


class ImageDiffusionInpaintWorkflow(WorkflowBuilder):

    def __init__(self):
        super().__init__("Image Diffusion - Inpaint", Image.Image, workflow=True, subworkflow=False)

    def build(self):
        image_input = ImageUploadInputNode()
        inpaint_scale_input = FloatUserInputNode(value = 1.0, name = "inpaint_scale")
        models_input = DiffusionModelUserInputNode(modeltype = "inpaint")
        loras_input = LORAModelUserInputNode(diffusion_model_input = models_input, name = "lora")
        prompt_input = TextAreaInputNode(value = "", name="prompt")
        negprompt_input = StringUserInputNode(value = "", name="negprompt")
        seed_input = SeedUserInputNode(value = None, name="seed")
        steps_input = IntUserInputNode(value = 20, name = "steps")
        cfgscale_input = FloatUserInputNode(value = 7.0, name = "cfgscale")
        scheduler_input = ListSelectUserInputNode(value = "DPMSolverMultistepScheduler", options = ImageDiffusionNode.SCHEDULERS, name="scheduler")

        init_condition = ConditioningInputNode(image = image_input, model = 'initimage', scale = inpaint_scale_input)
        maskimage = ImageAlphaToMaskNode(image = image_input, smooth = False, name = "maskimage")
        mask_condition = ConditioningInputNode(image = maskimage, model = 'mask_condition')

        prompt_processor = RandomPromptProcessorNode(prompt = prompt_input, name = "prompt_processor")
        models_input.addUpdateListener(lambda: prompt_processor.setWildcardDict(DiffusersPipelines.pipelines.getEmbeddingTokens(models_input.basemodel)))
        image_diffusion = ImageDiffusionNode(models = models_input,
                                    loras = loras_input,
                                    prompt = prompt_processor,
                                    negprompt = negprompt_input,
                                    steps = steps_input,
                                    cfgscale = cfgscale_input,
                                    seed = seed_input,
                                    scheduler = scheduler_input,
                                    conditioning_inputs = [ init_condition, mask_condition ])
        
        image_composite = ImageCompositeNode(foreground = image_diffusion, background = image_input, mask = maskimage, name = "composite")

        # TODO differential image to image pass
        
        return image_composite